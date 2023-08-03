"""Multilayer Perceptron (MLP) Regressors
"""

from torch import nn
from abcd.models.regression.Regressor import Regressor


class MLPReg(Regressor):
    def __init__(self, *args, input_size=28, **kwargs):
        super(MLPReg, self).__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.linear_layers = None

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_layers(x)
        return output.float()


class MLPReg3(MLPReg):
    def __init__(self, *args, **kwargs):
        super(MLPReg3, self).__init__(*args, **kwargs)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),   # output size is 1 for regression
        )