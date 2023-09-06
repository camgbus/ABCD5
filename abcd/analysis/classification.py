"""A module for training classification models.
"""

import os
import importlib
from collections import OrderedDict
import pandas as pd
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import shap
import matplotlib.pyplot as plt
import abcd.data.VARS as VARS
from abcd.data.read_data import get_subjects_events_visits, subject_cols_to_events
from abcd.data.define_splits import SITES, save_restore_visit_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.var_tailoring.discretization import discretize_var
from abcd.data.var_tailoring.residualization import residualize
from abcd.data.pytorch.get_dataset import PandasDataset
from abcd.data.pytorch.get_dataloaders import get_train_dl, get_eval_dls
from abcd.training.ClassifierTrainer import ClassifierTrainer
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def classification(exp, eval_config=False):
    '''Train a classifier within an experiment.
    '''
    config = exp.config
    
    # Fetch subjects and events
    visits = config.get("visits", ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1'])
    subjects_df, events_df = get_subjects_events_visits(visits)
    print(f"There are {len(events_df)} visits for {len(subjects_df)} subjects")
    target_col = config['target_col']
    
    
    # Rename and define columns
    events_df = events_df.rename(columns=VARS.CBCL_SCORES_t)
    subjects_df = subjects_df.rename(columns=VARS.NIH_TESTS_uncorrected)
    subjects_df = subjects_df.rename(columns=VARS.NIH_COMBINED_uncorrected)
    events_df = events_df.rename(columns=VARS.NAMED_CONNECTIONS)
    fmri_columns = list(VARS.NAMED_CONNECTIONS.values())
    # Filter repeated connections, as symmetric
    fmri_columns = ['-'.join(x) for x in set([tuple(sorted(x.split('-'))) for x in fmri_columns])]
    events_df = events_df.rename(columns=VARS.CONNECTIONS_C_SC)
    fmri_columns_subcor = list(VARS.CONNECTIONS_C_SC.values())
    smri_columns = [var_name + '_' + parcel for var_name in VARS.DESIKAN_STRUCT_FEATURES.keys() for parcel in VARS.DESIKAN_PARCELS[var_name] + VARS.DESIKAN_MEANS]
    nih_columns = list(VARS.NIH_TESTS_uncorrected.values()) + list(VARS.NIH_COMBINED_uncorrected.values())
    cbcl_columns = list(VARS.CBCL_SCORES_t.values())
    
    # Add environment and NIH columns to the events_df
    env_columns = ['kbi_sex_assigned_at_birth', 'race_ethnicity', 'site_id_l', 'demo_comb_income_v2']
    events_df = subject_cols_to_events(subjects_df, events_df, columns=env_columns+nih_columns)
    events_df = events_df.replace({'site_id_l': dict(zip([f"site{nr_id:02d}" for nr_id in range(1, 22)], range(0,21)))})
    env_columns += ["interview_age"]
            
    # If the target variable is continuous (over 25 possible values), discretize
    labels = sorted(list(set(events_df[target_col])))
    if len(labels) > 25:
        events_df = discretize_var(events_df, target_col, target_col+"_d", nr_bins=config.get("target_disc_bins", 4), by_freq=True)
        target_col = target_col+"_d"
        labels = sorted(list(set(events_df[target_col])), key=lambda x: float(x.replace("<= ", "")))
        
    # If stated in the configuration, binarize by only leaving the lower and upper groups
    if config.get("binary"):
        events_df = events_df.loc[((events_df[target_col] == labels[0])|(events_df[target_col] == labels[-1]))]
        labels = [labels[0], labels[-1]]
    print("Labels: {}".format(labels))
    print("There are {} visits".format(len(events_df)))

    # Change ABCD values to class integers starting from 0
    for ix, label in enumerate(labels):
        events_df.loc[events_df[target_col] == label, target_col] = ix
    labels = [VARS.VALUES[target_col][label] for label in labels] if target_col in VARS.VALUES else [str(label) for label in labels]
    events_df[target_col] = pd.to_numeric(events_df[target_col])

    # Print label distribution
    for val in set(events_df[target_col]):
        print('{} visits with {} target'.format(len(events_df.loc[events_df[target_col] == val]), labels[int(val)]))
        
    # Define features
    feature_cols = []
    if 'fmri' in config['features']:
        feature_cols += fmri_columns
    if 'fmri_sc' in config['features']:
        feature_cols += fmri_columns_subcor
    if 'smri' in config['features']:
        feature_cols += smri_columns

    # Normalize and residualize features
    residualization_columns = config.get("res_columns", ["kbi_sex_assigned_at_birth", "race_ethnicity", "smri_vol_cdk_mean"])
    print(f'Residualization with {residualization_columns}')
    for var_id in feature_cols:
        events_df = normalize_var(events_df, var_id, var_id)
        events_df = residualize(events_df, var_id, var_id, 
                    covs = residualization_columns,
                    references = {"kbi_sex_assigned_at_birth": 1.0, "race_ethnicity": 1.0, "site_id_l": 15})
        
    # Divide events by subject splits
    splits = save_restore_visit_splits(visits=visits, k=config.get("cv_k", 5))
    ood_site_id = SITES[config.get("cv_ood_ix", 0)]
    events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, splits, ood_site_id)
    print("Nr. events train: {}, val: {}, test: {}".format(len(events_train), len(events_id_test), len(events_ood_test)))
    
    # Define PyTorch datasets and dataloaders
    if config.get("divide_by_visit"):
        # Train only with the baseline visits, test on the remaining ones
        datasets = OrderedDict([
            ('Train', PandasDataset(events_train.loc[(events_train['eventname'] == 'baseline_year_1_arm_1')], feature_cols, target_col)),
            ('Train 2Y', PandasDataset(events_train.loc[(events_train['eventname'] == '2_year_follow_up_y_arm_1')], feature_cols, target_col)),
            ('ID Test B', PandasDataset(events_id_test.loc[(events_id_test['eventname'] == 'baseline_year_1_arm_1')], feature_cols, target_col)),
            ('ID Test 2Y', PandasDataset(events_id_test.loc[(events_id_test['eventname'] == '2_year_follow_up_y_arm_1')], feature_cols, target_col)),
            ('OOD Test B', PandasDataset(events_ood_test.loc[(events_ood_test['eventname'] == 'baseline_year_1_arm_1')], feature_cols, target_col)),
            ('OOD Test 2Y', PandasDataset(events_ood_test.loc[(events_ood_test['eventname'] == '2_year_follow_up_y_arm_1')], feature_cols, target_col))])
    else: 
        datasets = OrderedDict([
            ('Train', PandasDataset(events_train, feature_cols, target_col)),
            ('Val', PandasDataset(events_id_test, feature_cols, target_col)),
            ('Test', PandasDataset(events_ood_test, feature_cols, target_col))])
    
    # Determine device for training
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using {} device".format(device))
    
    # Create dataloaders
    batch_size = config['batch_size']
    train_dataloader = get_train_dl(datasets['Train'], batch_size, seed=SEED, device=device)
    eval_dataloaders = get_eval_dls(datasets, batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    # Define model
    models_path = os.path.join(exp.path, 'models')
    module = importlib.import_module(config['model'][0])
    model = getattr(module, config['model'][1])(save_path=models_path, labels=labels, input_size=len(feature_cols))
    model = model.to(device)
    print(model)
    
    # Define optimizer and trainer
    learning_rate = config['lr']
    if config.get('weighted'):
        class_weights = compute_class_weight('balanced', classes=range(len(labels)), y=datasets['Train'].y.numpy())
        class_weights = torch.tensor(class_weights,dtype=torch.float)
        print(f"Loss weights: {class_weights}")
        loss_f = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_f = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not eval_config: # Train model
        trainer_path = os.path.join(exp.path, 'trainer')
        trainer = ClassifierTrainer(trainer_path, device, optimizer, loss_f, labels=labels)
        nr_epochs = config['nr_epochs']
        trainer.train(model, train_dataloader, eval_dataloaders,
                    nr_epochs=nr_epochs, starting_from_epoch=0,
                    print_loss_every=int(nr_epochs/10), eval_every=int(nr_epochs/10), export_every=int(nr_epochs/5), verbose=True)        
    else: # Only evaluate
        results_path = os.path.join(exp.path, 'results')
        trainer = ClassifierTrainer(results_path, device, optimizer, loss_f, labels=labels)
        for state_name in eval_config["states"]:
            model.restore(state_name)
            trainer.eval(model, eval_dataloaders, epoch_ix=state_name, verbose=True)
            trainer.export(model, state_name=state_name, only_results=True, verbose=False)
            if eval_config.get("shap"):
                e = shap.DeepExplainer(model, datasets["Train"].X)
                val_X = datasets["Val"].X[:50]
                shap_values = e.shap_values(val_X)
                shap.summary_plot(shap_values, val_X, feature_names=feature_cols, max_display=20, class_names=labels, show=False)
                plt.savefig(os.path.join(trainer.trainer_path, f"shap_{state_name}.png"))
            