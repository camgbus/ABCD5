import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
from abcd.local.paths import output_path
from abcd.data.read_data import get_subjects_events_sf, subject_cols_to_events, add_event_vars
import abcd.data.VARS as VARS
from abcd.data.define_splits import SITES, save_restore_sex_fmri_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.var_tailoring.discretization import discretize_var, boundaries_by_frequency
from abcd.data.pytorch.get_dataset import PandasDataset

#regresssion-specific imports
from abcd.training.RegressorTrainer import RegressorTrainer

CLASS_NAMES = {
    "sex": ['male', 'female'],
    "fluid intelligence norm": ['<= 0.43', '<= 0.519', '<= 0.595', '> 0.595'],
    "fluid intelligence unnormalized": ['<= 86.00', '<= 93.00', '<= 99.00', '> 99.00'] 
    
}

THRESHOLDS = { # note that values < first class or > last class are assigned to closest class
    "sex": [0, 0.5, 1.0], #2 classes
    "fluid intelligence norm": [0, 0.43, 0.519, 0.595, 1.0], #4 classes
    "fluid intelligence unnormalized": [0, 86.0, 93.0, 99.0, 131.0]
}


def train_model(model, device, config, experiment_title, dataloaders, verbose=True, bucketing_scheme=None, l2_lambda=0.0):
    # Define class labels and thresholds
    thresholds = None
    class_names = None
    if bucketing_scheme in CLASS_NAMES and bucketing_scheme in THRESHOLDS:
        thresholds = THRESHOLDS[bucketing_scheme]
        class_names = CLASS_NAMES[bucketing_scheme]
    
    # Define optimizer and trainer
    loss_f = nn.MSELoss()
    trainer_path = os.path.join(output_path, experiment_title, 'results')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=l2_lambda)
    trainer = RegressorTrainer(trainer_path, device, optimizer, loss_f, 
                                thresholds=thresholds, class_names=class_names)

    # Train model
    trainer.train(model, dataloaders['train'], dataloaders, 
                nr_epochs=config['nr_epochs'], starting_from_epoch=0,
                print_loss_every=10, eval_every=10, export_every=50, verbose=verbose)
    
    return trainer


def preprocess(target_col, features, ood_site_num=0, normalize_targets=True):
    """
    Preprocess the data for regression (currently only supports sex predictions) and set up
    PyTorch dataloaders.

    Args:
        target_col: str
        features: List[str]
        ood_site_num: int, determines which site to use for ood testing
        normalize_targets: bool, whether to normalize the targets btw. 0 and 1 (default: True)
    Returns:
        dataloaders: OrderedDict of PyTorch dataloaders
        events_train, events_id_test, events_ood_test: Pandas DataFrames
        feature_cols: List[str]
    """

    # Fetch subjects and events
    subjects_df, events_df = get_subjects_events_sf()
    print("There are {} subjects and {} visits with imaging".format(len(subjects_df), len(events_df)))

    # Leave only the baseline visits
    events_df = events_df.loc[(events_df['eventname'] == 'baseline_year_1_arm_1')]
    print("Leaving baseline visits, we have {} events\n".format(len(events_df)))

    # Add the target to the events df, if not there
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
    print("There are {} visits after adding the target and removing NAs".format(len(events_df)))

    # Discritize targets (and place in new column)
    thresholds = None
    if target_col != 'kbi_sex_assigned_at_birth':
        print("\nDiscretizing targets for optional accuracy metric")
        # events_df = discretize_var(events_df, target_col, target_col+"_d", nr_bins=4, by_freq=True)
        # target_col_discretized = target_col+"_d"
        # labels = sorted(list(set(events_df[target_col_discretized])), key=lambda x: float(x.replace("<= ", "")))
        # labels[-1] = "> {}".format(labels[-2].replace("<= ", "")) #the last label should be > the second to last boundary
        # print("Labels: {}".format(labels))
        thresholds = boundaries_by_frequency(list(events_df[target_col]), 4)[1:]
        class_labels = ['<= {}'.format(x) for x in thresholds if x != thresholds[-1]] + ['> {}'.format(thresholds[-2])]
        print("Class labels: {}".format(class_labels))

    # Define features
    features_fmri = list(VARS.NAMED_CONNECTIONS.keys())
    features_smri = [var_name + '_' + parcel for var_name in VARS.DESIKAN_STRUCT_FEATURES.keys() for parcel in VARS.DESIKAN_PARCELS[var_name] + VARS.DESIKAN_MEANS]
    feature_cols = []
    if 'fmri' in features:
        feature_cols += features_fmri
    if 'smri' in features:
        feature_cols += features_smri
    
    # Normalize features
    for var_id in feature_cols:
        events_df = normalize_var(events_df, var_id, var_id)

    # Divide events into training, validation and testing
    splits = save_restore_sex_fmri_splits(k=5)
    ood_site_id = SITES[ood_site_num]
    events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, splits, ood_site_id)
    print("\nCreated splits")
    print("Nr. events: {} train, {} val, {} test".format(len(events_train), len(events_id_test), len(events_ood_test)))

    # Does not change results but prevents long error about fragmented dataframes by copying them into contiguous memory.
    events_train = events_train.copy()
    events_id_test = events_id_test.copy()
    events_ood_test = events_ood_test.copy()

    #Normalize targets with respect to min and max of training set
    if normalize_targets:
        normalized_target_col = target_col + "_norm"
        train_min = events_train[target_col].min()
        train_max = events_train[target_col].max()
        events_train = normalize_var(events_train, target_col, normalized_target_col, y_min=train_min, y_max=train_max)
        events_id_test = normalize_var(events_id_test, target_col, normalized_target_col, y_min=train_min, y_max=train_max)
        events_ood_test = normalize_var(events_ood_test, target_col, normalized_target_col, y_min=train_min, y_max=train_max)
        print("\nNormalized targets with respect to min and max of training set.\nNew target column containing normalized data: {}\n".format(normalized_target_col))
        target_col = normalized_target_col

        #Normalize thresholds
        if thresholds:
            thresholds = [round((x-train_min)/(train_max-train_min), 3) for x in thresholds]
            print("Normalized thresholds: {}".format(thresholds))

    # Print class distribution
    if target_col == 'kbi_sex_assigned_at_birth':
        labels = ["Male", "Female"]
        print("train distribution:")
        for val in set(events_train['kbi_sex_assigned_at_birth_norm']):
            print('\t{} visits with {} target'.format(len(events_train.loc[events_train["kbi_sex_assigned_at_birth_norm"] == val]), labels[int(val)]))
        print("val distribution:")
        for val in set(events_id_test['kbi_sex_assigned_at_birth_norm']):
            print('\t{} visits with {} target'.format(len(events_id_test.loc[events_id_test["kbi_sex_assigned_at_birth_norm"] == val]), labels[int(val)]))
        print('test distribution:')
        for val in set(events_ood_test['kbi_sex_assigned_at_birth_norm']):
            print('\t{} visits with {} target'.format(len(events_ood_test.loc[events_ood_test["kbi_sex_assigned_at_birth_norm"] == val]), labels[int(val)]))

    # Define custom PandasDatasetFloats datasets
    datasets = OrderedDict([('train', PandasDataset(events_train, feature_cols, target_col, torch.float32)),
                ('val', PandasDataset(events_id_test, feature_cols, target_col, torch.float32)),
                ('test', PandasDataset(events_ood_test, feature_cols, target_col, torch.float32))])
    
    # Create dataloaders
    batch_size = 64

    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)

    dataloaders = OrderedDict([
        ('train', train_loader),
        ('val', val_loader),
        ('test', test_loader)
    ])

    print("\nCreated dataloaders")
    for X, y in dataloaders['train']:
        print(f"Shape and datatype of X: {X.shape}, {X.dtype}")
        print(f"Shape and datatype of y: {y.shape}, {y.dtype}")
        break

    return dataloaders, events_train, events_id_test, events_ood_test, feature_cols, thresholds