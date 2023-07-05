import numpy as np
from abcd.validation.metrics.classification import confusion_matrix, binarize, recall, precision
from sklearn import metrics

def text_metrics():
    labels = [0, 1, 2]
    y_true = np.array([2, 0, 2, 2, 0, 1, 0, 1, 2, 0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 0, 2, 1, 1, 2, 0, 0, 1, 1])

    cm = confusion_matrix(y_true, y_pred)
    assert np.array_equal(cm, np.array([[3, 2, 0], [0, 2, 1], [1, 2, 2]]))

    tn_fp_fn_tp = []
    for label in labels:
        y_true_l, y_pred_l = binarize(y_true, y_pred, label=label)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true_l, y_pred_l).ravel()
        tn_fp_fn_tp.append([tn, fp, fn, tp])
    assert tn_fp_fn_tp[0] == [7, 1, 2, 3]
    assert tn_fp_fn_tp[1] == [6, 4, 1, 2]
    assert tn_fp_fn_tp[2] == [7, 1, 3, 2]

    # Accuracy = sum of TP for each class / total nr. elements
    accuracy = metrics.accuracy_score(y_true, y_pred)
    assert abs(accuracy-(7/13)) < 0.01

    for label in labels:
        p = tn_fp_fn_tp[label][3] + tn_fp_fn_tp[label][2]
        if p > 0:
            # Sensitivity = Recall = Hit rate = TPR = TP/P
            recall_score = recall(y_true, y_pred, binarize_by_label=label)
            manual_recall = tn_fp_fn_tp[label][3] / p
            assert abs(recall_score-manual_recall) < 0.01
            # Precision = Positive Predicted Value (PPV) = TP/(TP+FP)
            precision_score = precision(y_true, y_pred, binarize_by_label=label)
            manual_precision = tn_fp_fn_tp[label][3] / (tn_fp_fn_tp[label][3] + tn_fp_fn_tp[label][1])
            assert abs(precision_score-manual_precision) < 0.01