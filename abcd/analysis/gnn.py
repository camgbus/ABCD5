import os
import numpy as np
from collections import OrderedDict
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from abcd.data.pytorch.get_dataset import PandasDataset
import abcd.data.VARS as VARS
from abcd.data.read_data import get_subjects_events_sf, subject_cols_to_events, add_event_vars
from abcd.data.define_splits import SITES, save_restore_sex_fmri_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.var_tailoring.discretization import discretize_var
import importlib
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

#gnn specific imports
import networkx as nx
import matplotlib.patches as mpatches
from torch_geometric.data import Data


def visualize_graph(G, config, title=None):
    '''
    Visualize a graph using networkx and matplotlib.

    Args:
        G (networkx.Graph): undirected, weighted graph
        config (dict): configuration dictionary
        title (str): title of the plot (default: None)
    '''
    # Set up color map based on node type
    color_key = {"gordon network": "yellow"}
    if 'fmri_subcortical' in config['features']:
        color_key["subcortical"] = "#8dd3c7"

    color_map = []
    for node, attr in G.nodes(data=True):
        color_map.append(color_key[attr['type']])

    # Prepare the edge weights (line thickness corresponds to edge weight)
    edge_weights = []
    for (node1, node2, data) in G.edges(data=True):
        edge_weights.append(data['weight'])
    edge_weights = [w / max(edge_weights) * 2 for w in edge_weights]  # normalize edge weights to the range [0, 5]

    # Set size and layout
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=color_map, ax=ax, node_size=800, font_size=12, width=edge_weights)

    # Create legend on the upper right corner
    legend_elements = [mpatches.Patch(facecolor=color, label=label) for label, color in color_key.items()]
    plt.legend(handles=legend_elements, loc='upper right', prop={'size': 12})

    # Set title
    if title:
        plt.title(title, fontsize=16)
        
    # Show the plot
    plt.show()


def construct_networkx_graph(config, events_df, row_idx):
    '''
    Given events_df and a row index, construct a graph for the subject at that row.
    Nodes are gordon networks (as well as subcortical regions, if config['features'] contains
    'fmri_subcortical'). Weighted, undirected edges between nodes are the rsfmri (resting state) 
    connections.
    
    Args:
        config (dict): configuration dictionary
        events_df (pandas.DataFrame): dataframe with events
        row_idx (int): row index in events_df representing a particular subject
    Returns:
        G (networkx.Graph): undirected, weighted graph
    '''

    # initialize undirected graph
    G = nx.Graph()

    # add nodes
    for network in VARS.NETWORKS:
        G.add_node(network, type='gordon network')
    if 'fmri_subcortical' in config['features']:
        for region in VARS.SUBCORTICAL:
            G.add_node(region, type='subcortical')
    
    # add edges
    for col_name, val in events_df.loc[row_idx].items():
        if "rsfmri_c_ngd_" in col_name: #gordon network -> gordon network
            col_name = col_name[len("rsfmri_c_ngd_"):]
            region1, region2 = col_name.split("_ngd_")
            if not G.has_edge(region1, region2): #avoid duplicates (symmetric features)
                G.add_edge(region1, region2, weight=val)
        elif "rsfmri_cor_ngd_" in col_name: #gordon network -> subcortical region      
            col_name = col_name[len("rsfmri_cor_ngd_"):]
            region1, region2 = col_name.split("_scs_")
            region1_translated = VARS.NETWORK_NAMES_ASAG_to_NETWORK[region1]
            if not G.has_edge(region1, region2):
                G.add_edge(region1_translated, region2, weight=val)
    
    return G


def construct_pyg_datapoint(config, events_df, row_idx):
    """
    1. Constructs a PyTorch Geometric graph for the subject at the given row_idx. 
    Nodes include gordon networks. If config['features'] contains 'fmri_subcortical', then
    nodes will also include subcortical regions. Weighted edges are rsfmri (resting state 
    fmri) connections.
    2. Uses the graph to construct a PyTorch Geometric Data object for the subject at the given row_idx, 
    where y is a classification for the whole graph.

    Args:
        config (dict): configuration dictionary
        events_df (pandas.DataFrame): dataframe with events
        row_idx (int): row index in events_df representing a particular subject
    Returns:
        data (Data): PyTorch Geometric Data object representing the graph
    """
    if config['features'] not in [['fmri'], ['fmri', 'fmri_subcortical']]:
        raise ValueError("Unsupported config['features']:", config['features'])

    node_features = [] #(num_nodes, num_node_features)
    edge_index = [] #(2, num_edges) - graph connectivity in COO format with type torch.long
    edge_features = [] #(num_edges, num_edge_features)
    graph_label = events_df.loc[row_idx][config['target_col']] #(1, *), where * is shape of target; (1, 1) in this case

    # populate node_features
    for network in VARS.NETWORKS:
        node_features.append(network)
    if 'fmri_subcortical' in config['features']:
        for region in VARS.SUBCORTICAL:
            node_features.append(region)
            
    # populate edge_index and edge_features
    for col_name, val in events_df.loc[row_idx].items():
        region1, region2 = None, None
        subcortical_connection = False

        if "rsfmri_c_ngd_" in col_name:
            col_name = col_name[len("rsfmri_c_ngd_"):]
            region1, region2 = col_name.split("_ngd_")
        elif "rsfmri_cor_ngd_" in col_name:
            col_name = col_name[len("rsfmri_cor_ngd_"):]
            region1, region2 = col_name.split("_scs_")
            region1 = VARS.NETWORK_NAMES_ASAG_to_NETWORK[region1] #translate to gordon network name
            subcortical_connection = True
        
        if region1 in node_features and region2 in node_features:
            # Note: In PyTorch Geometric, there is no explicit "undirected graph" setting. Instead, undirected 
            # graphs are represented by adding both directions for each edge in the edge_index tensor. 
            # The "rsfmri_c_ngd_" table (non-subcortical) has symmetric entries, so two directed edges 
            # will be added btw. each region, which is desired for undirected grpahs.
            # However, the connections between gordon networks and subcortical regions are not symmetric, so we 
            # must manually add 2 edges at each subcortical connection.
            if region1 != region2:
                region1_node_idx = node_features.index(region1)
                region2_node_idx = node_features.index(region2)

                edge_index.append([region1_node_idx, region2_node_idx])
                edge_features.append((region1, region2, val))

                if subcortical_connection:
                    edge_index.append([region2_node_idx, region1_node_idx])
                    edge_features.append((region2, region1, val))
            else:
                region_node_idx = node_features.index(region1)
                edge_index.append([region_node_idx, region_node_idx])
                edge_features.append((region1, region2, val))

    # Check that the number of nodes and edges is correct
    assert len(edge_index) == len(edge_features)
    num_gordon_nodes = len(VARS.NETWORKS)
    num_subcortical_nodes = len(VARS.SUBCORTICAL)
    if config['features'] == ['fmri']:
        assert len(node_features) == num_gordon_nodes
        # each gordon node connects to itself and every other gordon node.
        assert len(edge_index) == num_gordon_nodes**2
    elif config['features'] == ['fmri', 'fmri_subcortical']:
        assert len(node_features) == num_gordon_nodes + num_subcortical_nodes
        # each gordon node connects to itself and every other gordon node.
        # each subcortical node connects to every gordon node, and there are 2 edges for each connection (bc undirected).
        assert len(edge_index) == (num_gordon_nodes**2) + (num_subcortical_nodes*num_gordon_nodes*2)
    
    # Create PyTorch Geometric Data object
    datapoint = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=graph_label)
    return datapoint

def generate_pyg_dataset(config, events_df):
      '''Given events_df, construct a PyTorch Geometric dataset.'''


def preprocess_data(config, ood_site_num=0):
    '''
    Preprocess the data for classification task and set up PyTorch dataloaders.

    Args:
        config (dict): configuration dictionary
        ood_site_num (int): which site to use for ood testing
    Returns:
        dataloaders (OrderedDict): OrderedDict of PyTorch dataloaders (contains train, val, test dataloaders)
        events_train, events_id_test, events_ood_test (pandas.DataFrame): dataframes with events
        labels (List[str]): class labels
        feature_cols (List[str]): feature columns
    '''
     
    # Fetch subjects and events
    subjects_df, events_df = get_subjects_events_sf()
    print("There are {} subjects and {} visits with imaging".format(len(subjects_df), len(events_df)))

    # Leave only the baseline visits
    events_df = events_df.loc[(events_df['eventname'] == 'baseline_year_1_arm_1')]
    print("Leaving baseline visits, we have {} events\n".format(len(events_df)))

    # Add the target to the events df, if not there
    target_col = config['target_col']
    if target_col not in events_df.columns:
        if target_col in subjects_df.columns:
            events_df = subject_cols_to_events(subjects_df, events_df, columns=[target_col])
        elif 'nihtbx' in target_col:
            events_df = add_event_vars(events_df, VARS.NIH_PATH, vars=[target_col])
        elif 'cbcl' in target_col:
            events_df = add_event_vars(events_df, VARS.CBCL_PATH, vars=[target_col])
        else:
            raise("Column {}, meant to be the target, was not recognized".format(target_col))
    events_df = events_df.dropna()
    print("There are {} visits after adding the target and removing NAs\n".format(len(events_df)))

    # If the target variable is continuous (over 25 possible values), discretize
    labels = sorted(list(set(events_df[target_col])))
    if len(labels) > 25:
        events_df = discretize_var(events_df, target_col, target_col+"_d", nr_bins=4, by_freq=True)
        target_col = target_col+"_d"
        labels = sorted(list(set(events_df[target_col])), key=lambda x: float(x.replace("<= ", "")))
    print("Labels: {}\n".format(labels))

    # Change ABCD values to class integers starting from 0
    for ix, label in enumerate(labels):
        events_df.loc[events_df[target_col] == label, target_col] = ix
    labels = [VARS.VALUES[target_col][label] for label in labels] if target_col in VARS.VALUES else [str(label) for label in labels]
    events_df[target_col] = pd.to_numeric(events_df[target_col])

    # Print label distribution
    for val in set(events_df[target_col]):
        print('{} visits with {} target'.format(len(events_df.loc[events_df[target_col] == val]), labels[int(val)]))
    print("\n")

    # Define features
    features_fmri = list(VARS.NAMED_CONNECTIONS.keys()) #gordon network -> gordon network
    features_fmri_subcortical = list(VARS.CONNECTIONS_C_SC) #gordon network -> subcortical region 
    features_smri = [var_name + '_' + parcel for var_name in VARS.DESIKAN_STRUCT_FEATURES.keys() for parcel in VARS.DESIKAN_PARCELS[var_name] + VARS.DESIKAN_MEANS]
    feature_cols = []
    if 'fmri' in config['features']:
        print("adding fmri gordon network features")
        feature_cols += features_fmri
    if 'fmri_subcortical' in config['features']:
        print("adding fmri subcortical features")
        feature_cols += features_fmri_subcortical
    if 'smri' in config['features']:
        print("\nadding smri features")
        feature_cols += features_smri

    # Normalize features
    for var_id in feature_cols:
        events_df = normalize_var(events_df, var_id, var_id)

    # Divide events into training, validation and testing
    splits = save_restore_sex_fmri_splits(k=5)
    ood_site_id = SITES[0]
    events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, splits, ood_site_id)
    print("\nNr. events train: {}, val: {}, test: ".format(len(events_train), len(events_id_test), len(events_ood_test)))

    # Define PyTorch datasets and dataloaders
    datasets = OrderedDict([('Train', PandasDataset(events_train, feature_cols, target_col)),
                ('Val', PandasDataset(events_id_test, feature_cols, target_col)),
                ('Test', PandasDataset(events_ood_test, feature_cols, target_col))])

    # Create dataloaders
    batch_size = config['batch_size']
    dataloaders = OrderedDict([(dataset_name, DataLoader(dataset, batch_size=batch_size, shuffle=True))
        for dataset_name, dataset in datasets.items()])

    for X, y in dataloaders['Train']:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    relevant_cols = feature_cols + [target_col, 'src_subject_id', 'eventname']
    return dataloaders, events_train[relevant_cols], events_id_test[relevant_cols], events_ood_test[relevant_cols], labels, feature_cols