"""Metrics for evaluating regression performance. Also includes a method to translate regression predictions to
class predictions given a list of thresholds and corresponding labels.

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

def translate_to_classes(y, class_labels, thresholds):
    '''Translates regression predictions to class predictions given a list of thresholds and corresponding labels.

    Params:
        y: numpy aray of shape (n, 1)
        class_labels: list of class labels
        thresholds: list of thresholds corresponding to each class. thresholds[i] is the lower bound of class i.
    Returns y_classes: numpy array of shape (n, 1) with class labels
    '''
    assert len(class_labels) == len(thresholds)
    y_classes = np.zeros(y.shape)
    for i, threshold in enumerate(thresholds):
        y_classes[y >= threshold] = class_labels[i]
    return y_classes