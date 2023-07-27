"""A general class for PyTorch classifiers.
"""

import torch
from torch import nn
from abcd.models.Model import Model

class Classifier(Model):
    def __init__(self, *args, labels=[], **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.labels = labels
        self.softmax = nn.Softmax(dim=1)

    def predict(self, X):
        '''Make a prediction based on a given input'''
        self.eval()
        with torch.no_grad():
            pred = self(X)
            return int(pred.argmax().detach())
        
    def predict_label(self, X):
        return self.labels[self.predict(X)]