"""A module for performing a CCA analysis.
"""
from collections import OrderedDict
import numpy as np
import pygal
import pandas as pd
from sklearn import metrics
import abcd.data.VARS as VARS
from abcd.data.read_data import get_subjects_events_visits, subject_cols_to_events
from abcd.analysis.methods.sklearn.sklearn_cca import fit_cca, add_cca_covariates, add_cca_preds, get_loadings
from abcd.data.define_splits import SITES, save_restore_visit_splits
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.data.var_tailoring.normalization import normalize_var
from abcd.data.var_tailoring.residualization import residualize
from abcd.plotting.seaborn.correlations import plot_correlations
from abcd.plotting.pygal.rendering import save as pygal_save
from abcd.utils.io import dump_df
from pygal.style import Style
from abcd.plotting.seaborn.rendering import save as plt_save
from abcd.plotting.pycirclize.plot_connections import plot_network_connections

def cca(exp):
    '''Train a CCA model.
    '''
    config = exp.config
    
    # Fetch subjects and events
    visits = config.get("visits", ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1'])
    subjects_df, events_df = get_subjects_events_visits(visits)
    print(f"There are {len(events_df)} visits for {len(subjects_df)} subjects")
    target_columns = config['target_columns']
    
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
        
    # Plot target columns
    target_columns = config['target_columns']
    custom_style = Style(colors=VARS.CBCL_colors)
    plot = pygal.Box(style=custom_style)
    for var_id in target_columns:
        plot.add(var_id, list(events_df[var_id]))
    pygal_save(plot, exp.path, file_name="targets_distribution")
        
    # Normalize target columns
    for var_id in target_columns:
        events_df = normalize_var(events_df, var_id, var_id)

    # Divide events by subject splits
    splits = save_restore_visit_splits(visits=visits, k=config.get("cv_k", 5))
    ood_site_id = SITES[config.get("cv_ood_ix", 0)]
    events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, splits, ood_site_id)
    print("Nr. events train: {}, val: {}, test: {}".format(len(events_train), len(events_id_test), len(events_ood_test)))
    
    # Define datasets
    if config.get("divide_by_visit"):
        # Train only with the baseline visits, test on the remaining ones
        datasets = OrderedDict([
            ('Train', events_train.loc[(events_train['eventname'] == 'baseline_year_1_arm_1')]), 
            ('Train 2Y', events_train.loc[(events_train['eventname'] == '2_year_follow_up_y_arm_1')]), 
            ('ID Test B', events_id_test.loc[(events_id_test['eventname'] == 'baseline_year_1_arm_1')]), 
            ('ID Test 2Y', events_id_test.loc[(events_id_test['eventname'] == '2_year_follow_up_y_arm_1')]), 
            ('OOD Test B', events_ood_test.loc[(events_ood_test['eventname'] == 'baseline_year_1_arm_1')]), 
            ('OOD Test 2Y', events_ood_test.loc[(events_ood_test['eventname'] == '2_year_follow_up_y_arm_1')])])
    else: 
        datasets = OrderedDict([('Train', events_train), ('Val', events_id_test), ('Test', events_ood_test)])
    
    # Fit CCA model
    nr_components = config.get("nr_components", 3)
    cca_model, datasets['Train'] = fit_cca(datasets['Train'].copy(), feature_cols, target_columns, nr_components=nr_components)
    
    # Make a selection of features, and re-fit
    nr_features = config.get("nr_features")
    if nr_features:
        x_loadings, y_loadings = get_loadings(cca_model)
        feature_cols_1 = sorted(feature_cols, key=lambda x: np.absolute(x_loadings[:,0][feature_cols.index(x)]), reverse=True)[:nr_features]
        feature_cols_2 = sorted(feature_cols, key=lambda x: np.absolute(x_loadings[:,1][feature_cols.index(x)]), reverse=True)[:nr_features]
        feature_cols_3 = sorted(feature_cols, key=lambda x: np.absolute(x_loadings[:,2][feature_cols.index(x)]), reverse=True)[:nr_features]
        feature_cols = list(set(feature_cols_1 + feature_cols_2 + feature_cols_3))

        cca_model, datasets['Train'] = fit_cca(datasets['Train'].copy(), feature_cols, target_columns, nr_components=nr_components)
    
    # Add covariates to the data frames
    for ds_name in datasets.keys():
        print(f"Dataset: {ds_name}")
        datasets[ds_name] = add_cca_covariates(datasets[ds_name].copy(), cca_model, feature_cols, target_columns, nr_components=nr_components)
        datasets[ds_name] = add_cca_preds(datasets[ds_name].copy(), cca_model, feature_cols, target_columns)

    # Print loadings connectome
    x_loadings, y_loadings = get_loadings(cca_model)
    for dim in range(nr_components):
        plot = plot_network_connections(connections=feature_cols, values=x_loadings[:,dim])
        plt_save(plot, path=exp.path, file_name=f"X_loadings_{dim}")
        y_loadings_dim = list(y_loadings[:,dim])
        direction_colors = ["#CD8E90" if x>0 else "#86AAD5" for x in y_loadings_dim]
        custom_style = Style(colors=tuple(direction_colors), value_font_size=20)
        plot = pygal.Bar(style=custom_style)
        for var_ix, var_id in enumerate(target_columns):
            plot.add(var_id, abs(y_loadings_dim[var_ix]))
        pygal_save(plot, exp.path, file_name=f"Y_loadings_{dim}")
        
    # Print correlation plots
    cc_x_columns, cc_y_columns = [f"CC{dim+1}_X" for dim in range(nr_components)], [f"CC{dim+1}_Y" for dim in range(nr_components)]
    for ds_name, ds_df in datasets.items():
        
        # Plot correlations as a circular connectome plot
        for x_col in cc_x_columns:
            corrs = [np.corrcoef(ds_df[x_col], ds_df[f])[0][1] for f in feature_cols]
            plot = plot_network_connections(connections=feature_cols, values=corrs)
            plt_save(plot, path=exp.path, file_name=f"{x_col}_{ds_name}")
        
        # Plot correlations as a barplot
        for y_col in cc_y_columns:
            corrs = [np.corrcoef(ds_df[y_col], ds_df[f])[0][1] for f in target_columns]
            direction_colors = ["#CD8E90" if x>0 else "#86AAD5" for x in corrs]
            custom_style = Style(colors=tuple(direction_colors), value_font_size=20)
            plot = pygal.Bar(style=custom_style)
            for var_ix, var_id in enumerate(target_columns):
                plot.add(var_id, abs(corrs[var_ix]))
            pygal_save(plot, exp.path, file_name=f"{y_col}_{ds_name}")
            
    # Evaluate by outputting the correlations for each dimension
    results = []
    for ds_name, ds_df in datasets.items():
        for dim in range(nr_components):
            results.append([ds_name, f"Corr{dim+1}", np.corrcoef(ds_df[f"CC{dim+1}_X"], ds_df[f"CC{dim+1}_Y"])[0][1]])
            
        for target in target_columns:
            results.append([ds_name, f"rMSE {target}", np.sqrt(metrics.mean_squared_error(ds_df[target], ds_df[target+'_pred']))])
            
    results = pd.DataFrame(results, columns=['Dataset', 'Metric', 'Value'])
    dump_df(results, exp.path, file_name='results')