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

def discretize(y, class_names, thresholds):
    """
    Translate regression predictions into class predictions.
    Params:
        y: list of floats
        class_names: list of class labels
        thresholds: list of bin thresholds. thresholds[i] is the lower bound of class i. thresholds[i+1] is the upper bound of class i.
    Return: y_classes: list of class labels
    """
    assert len(class_names) + 1 == len(thresholds)
    bin_indices = np.digitize(y, thresholds) - 1
    bin_indices = np.clip(bin_indices, 0, len(class_names)-1) #account for values below the first threshold or above the last threshold
    y_classes = [class_names[i] for i in bin_indices]
    return y_classes

def accuracy(y_true, y_pred, class_names, thresholds):
    '''Accuracy after translating predictions to classes

    Params:
        y_true, y_pred: List
        class_labels: list of class labels
        thresholds: list of thresholds corresponding to each class. thresholds[i] is the lower bound of class i.
    '''
    y_pred_classes = discretize(y_pred, class_names, thresholds)
    y_true_classes = discretize(y_true, class_names, thresholds)
    return metrics.accuracy_score(y_true_classes, y_pred_classes)