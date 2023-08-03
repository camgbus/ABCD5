"""Metrics for evaluating regression performance.
"""

from sklearn import metrics
import numpy as np

def mean_squared_error(y_true, y_pred):
    '''Mean Squared Error (MSE) = average((y_true - y_pred)^2)'''
    return metrics.mean_squared_error(y_true, y_pred)

def root_mean_squared_error(y_true, y_pred):
    '''Root Mean Squared Error (RMSE) = sqrt(MSE)'''
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    '''Mean Absolute Error (MAE) = average(abs(y_true - y_pred))'''
    return metrics.mean_absolute_error(y_true, y_pred)

def r2_score(y_true, y_pred):
    '''R^2 (coefficient of determination) regression score function.'''
    return metrics.r2_score(y_true, y_pred)

def explained_variance_score(y_true, y_pred):
    '''Explained variance regression score function'''
    return metrics.explained_variance_score(y_true, y_pred)