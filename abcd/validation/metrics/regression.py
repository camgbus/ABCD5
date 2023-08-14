"""Metrics for evaluating regression performance.

Metrics:
- Mean Absolute Error (MAE): Measures the average magnitude of the errors between predicted and observed values.
- Mean Squared Error (MSE): Measures the average squared differences between predicted and observed values.
- Root Mean Squared Error (RMSE): Square root of MSE. It has the same unit as the quantity plotted on the vertical or Y-axis.
- R-squared (R2): Represents the proportion of the variance for the dependent variable that's explained by independent variables.
- Explained Variance Score: Measures the proportion to which a model accounts for the variation of the dependent variable.
- Median Absolute Error: Median of all absolute differences between the target and the prediction.
"""

from sklearn import metrics
import numpy as np

def mean_absolute_error(y_true, y_pred):
    '''Mean Absolute Error (MAE)'''
    return metrics.mean_absolute_error(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    '''Mean Squared Error (MSE)'''
    return metrics.mean_squared_error(y_true, y_pred)

def root_mean_squared_error(y_true, y_pred):
    '''Root Mean Squared Error (RMSE)'''
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    '''R-squared (Coefficient of Determination)'''
    return metrics.r2_score(y_true, y_pred)

def explained_variance_score(y_true, y_pred):
    '''Explained Variance Score'''
    return metrics.explained_variance_score(y_true, y_pred)

def median_absolute_error(y_true, y_pred):
    '''Median Absolute Error'''
    return metrics.median_absolute_error(y_true, y_pred)