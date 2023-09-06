"""Canonical Correlation Analysis with scikit-learn.
"""

from sklearn.cross_decomposition import CCA
import numpy as np

def fit_cca(df, feature_columns, target_columns, nr_components=3, verbose=True):
    '''Fits a CCA model and returns both the model and the dataframe with the canonical covariates'''
    
    if not nr_components:
        nr_components = min(len(feature_columns), len(target_columns))
        
    X = df[feature_columns]
    Y = df[target_columns]

    # Instantiate the Canonical Correlation Analysis
    cca_model = CCA(n_components=nr_components)

    # Fit the model
    cca_model.fit(X, Y)
    
    # Add canonical covariates to the data frame
    X_trans, Y_trans = cca_model.transform(X, Y)
    for dim in range(nr_components):
        cc_x = X_trans[:, dim]
        cc_y = Y_trans[:, dim]
        df.loc[:, "CC{}_X".format(dim+1)] = cc_x
        df.loc[:, "CC{}_Y".format(dim+1)] = cc_y
        corr = np.corrcoef(cc_x, cc_y)[0][1]
        if verbose:
            print("Covariate on dimension {0} has score {1:.2f} and correlation {2:.2f}".format(dim+1, cca_model.score(X, Y), corr))
    return cca_model, df

def add_cca_covariates(df, cca_model, feature_columns, target_columns, nr_components=3, verbose=True):
    
    if not nr_components:
        nr_components = min(len(feature_columns), len(target_columns))
        
    X = df[feature_columns]
    Y = df[target_columns]

    # Add canonical covariates to the data frame
    X_trans, Y_trans = cca_model.transform(X, Y)

    for dim in range(nr_components):
        cc_x = X_trans[:, dim]
        cc_y = Y_trans[:, dim]
        df.loc[:, "CC{}_X".format(dim+1)] = cc_x
        df.loc[:, "CC{}_Y".format(dim+1)] = cc_y
        corr = np.corrcoef(cc_x, cc_y)[0][1]
        if verbose:
            print("Covariate on dimension {0} has score {1:.2f} and correlation {2:.2f}".format(dim+1, cca_model.score(X, Y), corr))
        
    return df

def add_cca_preds(df, cca_model, feature_columns, target_columns):
    X = df[feature_columns]
    Y_hat = cca_model.predict(X)
    for target_ix, target_name in enumerate(target_columns):
        df.loc[:, target_name+"_pred"] = Y_hat[:, target_ix]
    return df

def get_loadings(cca_model):
    x_loadings, y_loadings = cca_model.x_loadings_, cca_model.y_loadings_
    return x_loadings, y_loadings
    