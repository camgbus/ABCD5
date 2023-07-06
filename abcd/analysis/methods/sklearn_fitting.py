"""Model wrapper functions for scikit-learn.
"""
from sklearn.metrics import mean_absolute_error, max_error
import pandas as pd
import numpy as np
from tqdm import tqdm
import pygal
import warnings
from abcd.local.paths import output_path
from abcd.data.define_splits import SITES
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.plotting.pygal.rendering import display_html
from abcd.validation.metrics.classification import balanced_accuracy, f1, confusion_matrix
from abcd.plotting.seaborn.confusion_matrix import plot_confusion_matrix
from abcd.plotting.seaborn.rendering import save

def within_range(values, min_val=0, max_val=1):
    return [max(min(x, max_val), min_val) for x in values]
    
def plot_results_one_site(events_train, events_id_test, events_ood_test, y_column, y_column_name, site_id):
    plot = pygal.XY(stroke=False, range=(0, 1))
    plot.title = "Predictions for {} in site {}".format(y_column_name, site_id)
    plot.x_title = 'Actual'
    plot.y_title = 'Predicted'
    plot.add('Train', list(zip(within_range(events_train[site_id][y_column]), within_range(events_train[site_id][y_column+"_pred"]))))
    plot.add('ID Test', list(zip(within_range(events_id_test[site_id][y_column]), within_range(events_id_test[site_id][y_column+"_pred"]))))
    plot.add('OOD Test', list(zip(within_range(events_ood_test[site_id][y_column]), within_range(events_ood_test[site_id][y_column+"_pred"]))))
    display_html(plot)
    
def plot_cm_one_site(events_train, events_id_test, events_ood_test, y_column, plot_title, site_id, labels=None):
    cm = confusion_matrix(events_train[site_id][y_column], events_train[site_id][y_column+"_pred"])
    plot = plot_confusion_matrix(cm, labels=labels)
    save(plot, output_path, file_name="Train "+plot_title)
    cm = confusion_matrix(events_id_test[site_id][y_column], events_id_test[site_id][y_column+"_pred"])
    plot = plot_confusion_matrix(cm, labels=labels)
    save(plot, output_path, file_name="ID Test "+plot_title)
    cm = confusion_matrix(events_ood_test[site_id][y_column], events_ood_test[site_id][y_column+"_pred"])
    plot = plot_confusion_matrix(cm, labels=labels)
    save(plot, output_path, file_name="OOD Test "+plot_title)
    
def set_model_preds(model, events_df, site_splits, feature_columns, y_column, site_id):
    '''Adds an additional y_column_pred column with the prediction results'''
    events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, site_splits, site_id)
    X_train, X_id_test, X_ood_test = events_train[feature_columns], events_id_test[feature_columns], events_ood_test[feature_columns]
    y_train = events_train[y_column]
    model.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events_train[y_column+"_pred"] = model.predict(X_train)
        events_id_test[y_column+"_pred"] = model.predict(X_id_test)
        events_ood_test[y_column+"_pred"] = model.predict(X_ood_test)
    return events_train, events_id_test, events_ood_test


def model_classification_results_df(sites_events_train, sites_events_id_test, sites_events_ood_test, y_column):
    scores_balanced_accuracy = {"Train": [], "ID Test": [], "OOD Test": []}
    scores_macro_f1 = {"Train": [], "ID Test": [], "OOD Test": []}    
    for site_id in tqdm(SITES):
        # Add scores
        scores_balanced_accuracy["Train"].append(balanced_accuracy(sites_events_train[site_id][y_column], sites_events_train[site_id][y_column+"_pred"]))
        scores_balanced_accuracy["ID Test"].append(balanced_accuracy(sites_events_id_test[site_id][y_column], sites_events_id_test[site_id][y_column+"_pred"]))
        scores_balanced_accuracy["OOD Test"].append(balanced_accuracy(sites_events_ood_test[site_id][y_column], sites_events_ood_test[site_id][y_column+"_pred"]))
        scores_macro_f1["Train"].append(f1(sites_events_train[site_id][y_column], sites_events_train[site_id][y_column+"_pred"]))
        scores_macro_f1["ID Test"].append(f1(sites_events_id_test[site_id][y_column], sites_events_id_test[site_id][y_column+"_pred"]))
        scores_macro_f1["OOD Test"].append(f1(sites_events_ood_test[site_id][y_column], sites_events_ood_test[site_id][y_column+"_pred"]))
    split_names = ["Train", "ID Test", "OOD Test"]
    results = pd.DataFrame({"split": split_names, 
                            "Balanced Acc. mean": [np.mean(scores_balanced_accuracy[sn]) for sn in split_names], 
                            "Balanced Acc. std": [np.std(scores_balanced_accuracy[sn]) for sn in split_names], 
                            "F1 (macro) mean": [np.mean(scores_macro_f1[sn]) for sn in split_names], 
                            "F1 (macro) std": [np.std(scores_macro_f1[sn]) for sn in split_names]})
    return results

def model_results_df(sites_events_train, sites_events_id_test, sites_events_ood_test, y_column):
    scores_mae = {"Train": [], "ID Test": [], "OOD Test": []}
    scores_me = {"Train": [], "ID Test": [], "OOD Test": []}    
    for site_id in tqdm(SITES):
        # Add scores
        scores_mae["Train"].append(mean_absolute_error(sites_events_train[site_id][y_column], sites_events_train[site_id][y_column+"_pred"]))
        scores_mae["ID Test"].append(mean_absolute_error(sites_events_id_test[site_id][y_column], sites_events_id_test[site_id][y_column+"_pred"]))
        scores_mae["OOD Test"].append(mean_absolute_error(sites_events_ood_test[site_id][y_column], sites_events_ood_test[site_id][y_column+"_pred"]))
        scores_me["Train"].append(max_error(sites_events_train[site_id][y_column], sites_events_train[site_id][y_column+"_pred"]))
        scores_me["ID Test"].append(max_error(sites_events_id_test[site_id][y_column], sites_events_id_test[site_id][y_column+"_pred"]))
        scores_me["OOD Test"].append(max_error(sites_events_ood_test[site_id][y_column], sites_events_ood_test[site_id][y_column+"_pred"]))
    split_names = ["Train", "ID Test", "OOD Test"]
    results = pd.DataFrame({"split": split_names, 
                            "MAE mean": [np.mean(scores_mae[sn]) for sn in split_names], 
                            "MAE std": [np.std(scores_mae[sn]) for sn in split_names], 
                            "Max. error mean": [np.mean(scores_me[sn]) for sn in split_names], 
                            "Max. error std": [np.std(scores_me[sn]) for sn in split_names]})
    return results