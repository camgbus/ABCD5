import os
import numpy as np
from collections import OrderedDict
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import importlib
from sklearn.utils.class_weight import compute_class_weight
import shap
import matplotlib.pyplot as plt
import abcd.data.VARS as VARS
from abcd.data.read_data import get_subjects_events_sf, subject_cols_to_events, add_event_vars
from abcd.data.define_splits import SITES, save_restore_sex_fmri_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.var_tailoring.discretization import discretize_var
from abcd.data.pytorch.get_dataset import PandasDataset
from abcd.training.ClassifierTrainer import ClassifierTrainer

def classification(exp, eval_config=False):
    
    config = exp.config
    
    # Fetch subjects and events
    subjects_df, events_df = get_subjects_events_sf()
    print("There are {} subjects and {} visits with imaging".format(len(subjects_df), len(events_df)))
    # Leave only the baseline visits
    events_df = events_df.loc[(events_df['eventname'] == 'baseline_year_1_arm_1')]
    print("Leaving baseline visits, we have {} visits".format(len(events_df)))

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
    print("There are {} visits after adding the target and removing NAs".format(len(events_df)))
        
    # If the target variable is continuous (over 25 possible values), discretize
    labels = sorted(list(set(events_df[target_col])))
    if len(labels) > 25:
        events_df = discretize_var(events_df, target_col, target_col+"_d", nr_bins=4, by_freq=True)
        target_col = target_col+"_d"
        labels = sorted(list(set(events_df[target_col])), key=lambda x: float(x.replace("<= ", "")))
    print("Labels: {}".format(labels))

    # Change ABCD values to class integers starting from 0
    for ix, label in enumerate(labels):
        events_df.loc[events_df[target_col] == label, target_col] = ix
    labels = [VARS.VALUES[target_col][label] for label in labels] if target_col in VARS.VALUES else [str(label) for label in labels]
    events_df[target_col] = pd.to_numeric(events_df[target_col])

    # Print label distribution
    for val in set(events_df[target_col]):
        print('{} visits with {} target'.format(len(events_df.loc[events_df[target_col] == val]), labels[int(val)]))
        
    # Define features
    features_fmri = list(VARS.NAMED_CONNECTIONS.keys())
    features_smri = [var_name + '_' + parcel for var_name in VARS.DESIKAN_STRUCT_FEATURES.keys() for parcel in VARS.DESIKAN_PARCELS[var_name] + VARS.DESIKAN_MEANS]
    feature_cols = []
    if 'fmri' in config['features']:
        feature_cols += features_fmri
    if 'smri' in config['features']:
        feature_cols += features_smri

    # Normalize features
    for var_id in feature_cols:
        events_df = normalize_var(events_df, var_id, var_id)
        
    # Divide events into training, validation and testing
    splits = save_restore_sex_fmri_splits(k=5)
    ood_site_id = SITES[0]
    events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, splits, ood_site_id)
    print("Nr. events train: {}, val: {}, test: {}".format(len(events_train), len(events_id_test), len(events_ood_test)))
    
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
    
    # Determine device for training
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using {} device".format(device))
    
    # Define model
    models_path = os.path.join(exp.path, 'models')
    module = importlib.import_module(config['model'][0])
    model = getattr(module, config['model'][1])(save_path=models_path, labels=labels, input_size=len(feature_cols))
    #model = FullyConnected5(save_path=models_path, labels=labels, input_size=len(feature_cols))
    model = model.to(device)
    print(model)
    
    # Define optimizer and trainer
    learning_rate = config['lr']
    if config.get('weighted'):
        class_weights = compute_class_weight('balanced', classes=range(len(labels)), y=datasets['Train'].y.numpy())
        class_weights = torch.tensor(class_weights,dtype=torch.float)
        print("Loss weights: {}".format(class_weights))
        loss_f = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_f = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not eval_config: # Train model
        trainer_path = os.path.join(exp.path, 'trainer')
        trainer = ClassifierTrainer(trainer_path, device, optimizer, loss_f, labels=labels)
        nr_epochs = config['nr_epochs']
        trainer.train(model, dataloaders['Train'], dataloaders, 
                    nr_epochs=nr_epochs, starting_from_epoch=0,
                    print_loss_every=int(nr_epochs/10), eval_every=int(nr_epochs/10), export_every=int(nr_epochs/5), verbose=True)        
    else: # Only evaluate
        results_path = os.path.join(exp.path, 'results')
        trainer = ClassifierTrainer(results_path, device, optimizer, loss_f, labels=labels)
        for state_name in eval_config["states"]:
            model.restore(state_name)
            trainer.eval(model, dataloaders, epoch_ix=state_name, verbose=True)
            trainer.export(model, state_name=state_name, only_results=True, verbose=False)
            if eval_config.get("shap"):
                e = shap.DeepExplainer(model, datasets["Train"].X)
                val_X = datasets["Val"].X[:50]
                shap_values = e.shap_values(val_X)
                shap.summary_plot(shap_values, val_X, feature_names=feature_cols, max_display=20, class_names=labels, show=False)
                plt.savefig(os.path.join(trainer.trainer_path, "shap_{}.png".format(state_name)))
            