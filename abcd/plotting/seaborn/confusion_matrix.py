"""A confusion matrix for classification tasks.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels=None, figure_size=(5.25,3.75), vmin=None, vmax=None, cmap=None):
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
    return plt

"""
class ConfusionMatrix:
    def __init__(self, cm, nr_classes, name_lst=None):
        self.cm = [[0 for i in range(nr_classes)] for i in range(nr_classes)]
        self.name_lst = name_lst

    def plot(self, label_predicted='Predicted', 
            label_actual='Actual', figure_size=(5.25,3.75), annot=True, 
            vmin=None, vmax=None, cmap=None, same_at_corner=True):
        
        nr_rows = len(cm)
        
        #if same_at_corner:
        #    # Invert the order of rows so the same item is at the corner (easier to read)
        #    cm.reverse()
            
        cm.insert(0, [0]*nr_rows)
        if self.name_lst is None:
            self.name_lst = [c+1 for c in range(nr_rows)]
        df = pd.DataFrame(cm, columns=self.name_lst)
        df = df.drop([0])

        if same_at_corner:
            self.name_lst.reverse()

        if self.name_lst is not None:
            df.index = self.name_lst
        plt.figure()
        sns.set(rc={'figure.figsize':figure_size})
        ax = sns.heatmap(df, annot=annot, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set(xlabel=label_predicted, ylabel=label_actual)
        return plt
"""