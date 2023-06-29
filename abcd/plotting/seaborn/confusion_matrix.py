"""A confusion matrix for classification tasks.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    def __init__(self, nr_classes, name_lst=None):
        self.cm = [[0 for i in range(nr_classes)] for i in range(nr_classes)]
        self.name_lst = name_lst

    def add(self, predicted, actual, count=1):
        if isinstance(predicted, str):
            self.cm[self.name_lst.index(actual)][self.name_lst.index(predicted)] += count
        else:
            self.cm[actual][predicted] += count

    def plot(self, path, name='confusion_matrix', label_predicted='Predicted', 
            label_actual='Actual', figure_size=(5.25,3.75), annot=True, 
            vmin=None, vmax=None, cmap=None, endings=['.png', '.svg'], same_at_corner=True):
        cm = self.cm.copy()
        nr_rows = len(cm)
        
        if same_at_corner:
            # Invert the order of rows so the same item is at the corner (easier to read)
            cm.reverse()
            
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
        for ending in endings:
            plt.savefig(os.path.join(path, name+ending), facecolor='w', bbox_inches="tight", dpi = 300)
        
    def get_accuracy(self):
        correct = sum([self.cm[i][i] for i in range(len(self.cm))])
        all_instances = sum([sum(x) for x in self.cm])
        return correct/all_instances