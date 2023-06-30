"""Model wrapper functions for scikit-learn.
"""
from sklearn.metrics import mean_absolute_error, max_error
import pandas as pd
import numpy as np
from tqdm import tqdm
import pygal
from abcd.data.define_splits import SITES
from abcd.data.divide_with_splits import divide_events_by_splits
from abcd.plotting.pygal.rendering import display_html

def plot_results_one_site(model, events_df, site_splits, feature_columns, y_column, y_column_name, site_id):
    events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, site_splits, site_id)
    X_train, X_id_test, X_ood_test = events_train[feature_columns], events_id_test[feature_columns], events_ood_test[feature_columns]
    y_train, y_id_test, y_ood_test = events_train[y_column], events_id_test[y_column], events_ood_test[y_column]
    model.fit(X_train, y_train)
    pred_train, pred_id_test, pred_ood_test = model.predict(X_train), model.predict(X_id_test), model.predict(X_ood_test)
    plot = pygal.XY(stroke=False)
    plot.title = y_column_name
    plot.x_title = 'Actual'
    plot.y_title = 'Predicted'
    plot.add('Train', list(zip(y_train, pred_train)))
    plot.add('ID Test', list(zip(y_id_test, pred_id_test)))
    plot.add('OOD Test', list(zip(y_ood_test, pred_ood_test)))
    display_html(plot)

def calculate_regession_results(model, events_df, site_splits, feature_columns, y_column):
    scores_mae = {"Train": [], "ID Test": [], "OOD Test": []}
    scores_me = {"Train": [], "ID Test": [], "OOD Test": []}
    for site_id in tqdm(SITES):
        events_train, events_id_test, events_ood_test = divide_events_by_splits(events_df, site_splits, site_id)
        X_train, X_id_test, X_ood_test = events_train[feature_columns], events_id_test[feature_columns], events_ood_test[feature_columns]
        y_train, y_id_test, y_ood_test = events_train[y_column], events_id_test[y_column], events_ood_test[y_column]
        model.fit(X_train, y_train)
        pred_train, pred_id_test, pred_ood_test = model.predict(X_train), model.predict(X_id_test), model.predict(X_ood_test)
        # Add scores
        scores_mae["Train"].append(mean_absolute_error(y_train, pred_train))
        scores_mae["ID Test"].append(mean_absolute_error(y_id_test, pred_id_test))
        scores_mae["OOD Test"].append(mean_absolute_error(y_ood_test, pred_ood_test))
        scores_me["Train"].append(max_error(y_train, pred_train))
        scores_me["ID Test"].append(max_error(y_id_test, pred_id_test))
        scores_me["OOD Test"].append(max_error(y_ood_test, pred_ood_test))  
    split_names = ["Train", "ID Test", "OOD Test"]
    results = pd.DataFrame({"split": split_names, 
                            "MAE mean": [np.mean(scores_mae[sn]) for sn in split_names], 
                            "MAE std": [np.std(scores_mae[sn]) for sn in split_names], 
                            "Max. error mean": [np.mean(scores_me[sn]) for sn in split_names], 
                            "Max. error std": [np.std(scores_me[sn]) for sn in split_names]})
    return results