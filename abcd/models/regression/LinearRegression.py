"""PyTorch linear regressor
"""

import torch
from torch import nn
from abcd.models.Model import Model

class LinearRegressor(Model):
    def __init__(self, *args, input_size, **kwargs):
        super(LinearRegressor, self).__init__(*args, **kwargs) 
        self.linear = nn.Linear(input_size, 1) #output size is 1

    def forward(self, x):
        '''Forward pass for linear regression model'''
        return self.linear(x).squeeze()

    def predict(self, X):
        '''Make a prediction based on a given input'''
        self.eval()  # set the model to evaluation mode
        with torch.no_grad():
            pred = self(X)
            return pred.detach()