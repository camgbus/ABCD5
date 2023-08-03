"""A general class for PyTorch regressors.
"""

import torch
from torch import nn
from abcd.models.Model import Model

class Regressor(Model):
    def __init__(self, *args, **kwargs):
        super(Regressor, self).__init__(*args, **kwargs)

    def predict(self, X):
        '''Make a prediction based on a given input'''
        self.eval()
        with torch.no_grad():
            pred = self(X)
            return pred.item()