import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
from abcd.local.paths import output_path
from abcd.data.read_data import get_subjects_events_sf, subject_cols_to_events
import abcd.data.VARS as VARS
from abcd.data.define_splits import SITES, save_restore_sex_fmri_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.pytorch.get_dataset import PandasDataset

#regresssion-specific imports
from abcd.models.regression.MLPRegressor import MLPRegressor, LinearRegressor, MLPRegressor3, MLPRegressorCustom
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from abcd.training.RegressorTrainer import RegressorTrainer
from abcd.validation.metrics.regression import translate_to_classes

#plotting
import matplotlib.pyplot as plt
import pygal
from abcd.plotting.pygal.rendering import display_html
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from abcd.plotting.seaborn.confusion_matrix import plot_confusion_matrix


def train_model(model, device, config, experiment_title, dataloaders, verbose=True):
    # Define optimizer and trainer
    loss_f = nn.MSELoss()
    trainer_path = os.path.join(output_path, experiment_title, 'results')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    trainer = RegressorTrainer(trainer_path, device, optimizer, loss_f)

    # Train model
    trainer.train(model, dataloaders['train'], dataloaders, 
                nr_epochs=config['nr_epochs'], starting_from_epoch=0,
                print_loss_every=10, eval_every=10, export_every=50, verbose=verbose)


def get_binary_class_accuracies(cm, class_names):
    TN, FP, FN, TP = cm.ravel()
    acc_0 = TN / (TN + FN) if (TN + FN) != 0 else 0
    acc_1 = TP / (TP + FP) if (TP + FP) != 0 else 0
    print(f"Accuracy for {class_names[0]} class: {acc_0:.2f}")
    print(f"Accuracy for {class_names[1]} class: {acc_1:.2f}")
    return acc_0, acc_1
    

def evaluate_regression(model, events_id_test, events_ood_test, target_col, feature_cols, experiment_title='', show_cm=True):
    #test dataframes
    X_id_test = events_id_test.loc[:, feature_cols]
    y_id_test = events_id_test[target_col]
    X_ood_test = events_ood_test.loc[:, feature_cols]
    y_ood_test = events_ood_test[target_col]

    #convert test dataframes to tensors
    X_id_test_tensor = torch.tensor(X_id_test.values, dtype=torch.float)
    y_id_test_tensor = torch.tensor(y_id_test.values, dtype=torch.float).view(-1, 1)
    X_ood_test_tensor = torch.tensor(X_ood_test.values, dtype=torch.float)
    y_ood_test_tensor = torch.tensor(y_ood_test.values, dtype=torch.float).view(-1, 1)

    #evaluate the model using the tensors
    model.eval()
    with torch.no_grad():
        y_id_pred_tensor = model(X_id_test_tensor)
        y_ood_pred_tensor = model(X_ood_test_tensor)
    
    #convert predictions to np arrays
    y_id_pred = y_id_pred_tensor.numpy()
    y_id_test = y_id_test_tensor.numpy()
    y_ood_pred = y_ood_pred_tensor.numpy()
    y_ood_test = y_ood_test_tensor.numpy()

    # Calculate the performance metrics: MAE, MSE, R2
    mae_id = mean_absolute_error(y_id_test, y_id_pred)
    mse_id = mean_squared_error(y_id_test, y_id_pred)
    # r2_id = r2_score(y_id_test, y_id_pred)
    mae_ood = mean_absolute_error(y_ood_test, y_ood_pred)
    mse_ood = mean_squared_error(y_ood_test, y_ood_pred)
    # r2_ood = r2_score(y_ood_test, y_ood_pred)

    #Translate to classes and generate confusion matrix
    class_labels = [0.0, 1.0]
    class_names = ['Male', 'Female']
    thresholds = [float('-inf'), 0.5]

    y_classes_id = translate_to_classes(y_id_pred, class_labels, thresholds)
    cm_id = confusion_matrix(y_id_test, y_classes_id)
    if show_cm:
        plot_confusion_matrix(cm_id, labels=class_names, title=experiment_title)
        print("\n")

    y_classes_ood = translate_to_classes(y_ood_pred, class_labels, thresholds)
    cm_ood = confusion_matrix(y_ood_test, y_classes_ood)
    if show_cm:
        plot_confusion_matrix(cm_ood, labels=class_names, title=experiment_title)
  
    if show_cm:
        print("id test")
        print("Mean absolute error (MAE): ", mae_id)
        print("Mean squared error (MSE): ", mse_id)
        # print("r2_score: ", r2_id)
    male_acc_id, female_acc_id = get_binary_class_accuracies(cm_id, class_names)

    if show_cm:
        print("\nood test")
        print("Mean absolute error (MAE): ", mae_ood)
        print("Mean squared error (MSE): ", mse_ood)
        # print("r2_score: ", r2_ood)
    male_acc_ood, female_acc_ood = get_binary_class_accuracies(cm_ood, class_names)

    return {
        'mae_id': mae_id,
        'mse_id': mse_id,
        'male_acc_id': male_acc_id,
        'female_acc_id': female_acc_id,
        'cm_id': cm_id,

        'mae_ood': mae_ood,
        'mse_ood': mse_ood,
        'male_acc_ood': male_acc_ood,
        'female_acc_ood': female_acc_ood,
        'cm_ood': cm_ood
    }


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

    #currently only supports sex predictions 
    #TODO: Generalize to any target column
    if target_col != 'kbi_sex_assigned_at_birth':
        print("Target column is not kbi_sex_assigned_at_birth. Exiting.")

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
    # Change kbi_sex_assigned_at_birth from 1.0 to 0.0 (male) and 2.0 to 1.0 (female)
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