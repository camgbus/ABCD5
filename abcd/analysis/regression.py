import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
from abcd.local.paths import output_path
from abcd.data.read_data import get_subjects_events_sf, subject_cols_to_events
import abcd.data.VARS as VARS
from abcd.data.define_splits import SITES, save_restore_sex_fmri_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.pytorch.get_dataset import PandasDataset

#regresssion-specific imports
from abcd.training.RegressorTrainer import RegressorTrainer

CLASS_NAMES = {"sex": ['male', 'female']}
THRESHOLDS = {"sex": [0.0, 0.5, 1.0]}


def train_model(model, device, config, experiment_title, dataloaders, verbose=True, bucketing_scheme=None):
    # Define class labels and thresholds
    thresholds = None
    class_names = None
    if bucketing_scheme:
        thresholds = THRESHOLDS[bucketing_scheme]
        class_names = CLASS_NAMES[bucketing_scheme]
    
    # Define optimizer and trainer
    loss_f = nn.MSELoss()
    trainer_path = os.path.join(output_path, experiment_title, 'results')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    trainer = RegressorTrainer(trainer_path, device, optimizer, loss_f, 
                                thresholds=thresholds, class_names=class_names)

    # Train model
    trainer.train(model, dataloaders['train'], dataloaders, 
                nr_epochs=config['nr_epochs'], starting_from_epoch=0,
                print_loss_every=10, eval_every=10, export_every=50, verbose=verbose)
    
    return trainer


def preprocess(target_col, features, ood_site_num=0):
    """
    Preprocess the data for regression (currently only supports sex predictions) and set up
    PyTorch dataloaders.

    Args:
        target_col: str
        features: List[str]
        ood_site_num: int, determines which site to use for ood testing
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
        events_df = subject_cols_to_events(subjects_df, events_df, columns=[target_col])

    #Normalize targets btw. 0 and 1
    if target_col == "kbi_sex_assigned_at_birth": # Change kbi_sex_assigned_at_birth from 1.0 to 0.0 (male) and 2.0 to 1.0 (female)
        events_df.loc[events_df["kbi_sex_assigned_at_birth"] == 1.0, "kbi_sex_assigned_at_birth"] = 0.0
        events_df.loc[events_df["kbi_sex_assigned_at_birth"] == 2.0, "kbi_sex_assigned_at_birth"] = 1.0

    # Print distribution
    labels = ["Male", "Female"]
    for val in set(events_df['kbi_sex_assigned_at_birth']):
        print('{} visits with {} target'.format(len(events_df.loc[events_df["kbi_sex_assigned_at_birth"] == val]), labels[int(val)]))

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

    return dataloaders, events_train, events_id_test, events_ood_test, feature_cols