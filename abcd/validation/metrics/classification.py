"""Metrics for evaluating classification performance.

Positives = P = TP + FN
Negatives = N = TN + FP

Metrics:
- Sensitivity = Recall = Hit rate = TPR = TP/P
- Specificity = TNR = TN/N
- FPR = FP/N
- Precision = Positive Predicted Value (PPV) = TP/(TP+FP)
- Negative Predicted Value (NPV) =  TN/(TN+FN)

Averaging modes for multi-class classification:
- binary: only report results for positive label
- micro: metrics are calculated by summing the individual TP, TN, FP, FN
- macro: metrics are calculated for each class, then averaged without weighting -> better for inbalanced data
- weighted: metrics are calculated for each class, then averaged by label frequency (i.e. w.r.t. support)
"""

from sklearn import metrics
import numpy as np

# Multi-class scores that are already averaged in a specific ways and receive lists of ints

def confusion_matrix(y_true, y_pred):
    '''Returns a confusion matrix, where the in the first dimension are the true classes and in the
    second, what they were predicted as. For instance cm[0] are the values for the true class 0'''
    return metrics.confusion_matrix(y_true, y_pred)

def accuracy(y_true, y_pred):
    '''Accuracy = sum of TP for each class / total nr. elements'''
    return metrics.accuracy_score(y_true, y_pred)

def balanced_accuracy(y_true, y_pred):
    '''Balanced accuracy = average of recalls for each class'''
    return metrics.balanced_accuracy_score(y_true, y_pred)

# Scores that receive probability outputs instead instead of predictions
# AUROC: Area under plotting TPR (y) against FPR (x)
#fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_model_2, pos_label=1)
#auc_model_2 = metrics.auc(fpr, tpr)
#metrics.roc_auc_score(y_true, y_probs, average=avg_mode)

# Brier = mean squared difference between the predicted probability and the actual outcome

# Scores that can be averaged in different ways

def f1(y_true, y_pred, avg_mode='macro', binarize_by_label=None, zero_div='warn'):
    '''F1 Score = 2TP/(2TP+FP+FN)'''
    if binarize_by_label is not None:
        y_true, y_pred = binarize(y_true, y_pred, label=binarize_by_label)
        avg_mode = 'binary'
    return metrics.f1_score(y_true, y_pred, average=avg_mode, zero_division=zero_div)

def recall(y_true, y_pred, avg_mode='macro', binarize_by_label=None, zero_div='warn'):
    '''Sensitivity = Recall = Hit rate = TPR = TP/P'''
    if binarize_by_label is not None:
        y_true, y_pred = binarize(y_true, y_pred, label=binarize_by_label)
        avg_mode = 'binary'
    return metrics.recall_score(y_true, y_pred, average=avg_mode, zero_division=zero_div)

def sensitivity(y_true, y_pred, avg_mode='macro', binarize_by_label=None, zero_div='warn'):
    return recall(y_true, y_pred, avg_mode, binarize_by_label, zero_div)

def precision(y_true, y_pred, avg_mode='macro', binarize_by_label=None, zero_div='warn'):
    '''Precision = Positive Predicted Value (PPV) = TP/(TP+FP)'''
    if binarize_by_label is not None:
        y_true, y_pred = binarize(y_true, y_pred, label=binarize_by_label)
        avg_mode = 'binary'
    return metrics.precision_score(y_true, y_pred, average=avg_mode, zero_division=zero_div)

def binarize(y_true, y_pred, label=1):
    ''''Leaves 1 if value == label, otherwise 0'''
    y_true_l = (y_true == label).astype(int)
    y_pred_l = (y_pred == label).astype(int)
    return y_true_l, y_pred_l