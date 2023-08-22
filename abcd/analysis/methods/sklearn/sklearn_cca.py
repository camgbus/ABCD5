"""Canonical Correlation Analysis with scikit-learn.
"""

from sklearn.cross_decomposition import CCA
import numpy as np

def fit_cca(df, feature_columns, target_columns, nr_components=0):
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
        df["CC{}_X".format(dim+1)] = cc_x
        df["CC{}_Y".format(dim+1)] = cc_y
        corr = np.corrcoef(cc_x, cc_y)[0][1]
        print("Covariate on dimension {0} has score {1:.2f} and correlation {2:.2f}".format(dim+1, cca_model.score(X, Y), corr))
    return cca_model, df

def add_cca_covariates(df, cca_model, feature_columns, target_columns, nr_components=0):
    
    if not nr_components:
        nr_components = min(len(feature_columns), len(target_columns))
        
    X = df[feature_columns]
    Y = df[target_columns]

    # Add canonical covariates to the data frame
    X_trans, Y_trans = cca_model.transform(X, Y)
    for dim in range(nr_components):
        cc_x = X_trans[:, dim]
        cc_y = Y_trans[:, dim]
        df["CC{}_X".format(dim+1)] = cc_x
        df["CC{}_Y".format(dim+1)] = cc_y
        corr = np.corrcoef(cc_x, cc_y)[0][1]
        print("Covariate on dimension {0} has score {1:.2f} and correlation {2:.2f}".format(dim+1, cca_model.score(X, Y), corr))