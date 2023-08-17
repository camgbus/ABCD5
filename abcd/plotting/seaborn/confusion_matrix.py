"""A confusion matrix for classification tasks.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels=None, figure_size=(5.25,3.75), vmin=None, vmax=None, cmap=None, title=''):
    '''Receives a confusion matrix where the in the first dimension are the true classes and in the
    second, what they were predicted as. For instance cm[0] are the values for the true class 0'''
    df = pd.DataFrame(cm, columns=labels)
    if labels is None:
        labels = [str(ix) for ix in range(len(cm))]
    df.index = labels
    plt.figure()
    sns.set(rc={'figure.figsize':figure_size})
    ax = sns.heatmap(df, annot=True, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set(xlabel='Predicted', ylabel='Actual')
    ax.set_title(title)
    return plt