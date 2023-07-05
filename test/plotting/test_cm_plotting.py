"""Tests whether a confusion matrix plot can be properly created.
"""

import os
import numpy as np
import pandas as pd
from abcd.validation.metrics.classification import confusion_matrix
from abcd.plotting.seaborn.confusion_matrix import plot_confusion_matrix
from abcd.plotting.seaborn.rendering import save

def test_plot_cm():
    output_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'test_objects'))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    y_true = np.array([2, 0, 2, 2, 0, 1, 0, 1, 2, 0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 0, 2, 1, 1, 2, 0, 0, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    plot = plot_confusion_matrix(cm, labels=["A", "B", "C"])
    save(plot, output_path, file_name="test_plot_cm")