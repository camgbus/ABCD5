"""Classifiers made excludively out of fully-connected layers.
"""

from torch import nn
from abcd.models.classification.Classifier import Classifier

class FullyConnected(Classifier):
    def __init__(self, *args, input_size=28, **kwargs):
        super(FullyConnected, self).__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.linear_layers = None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_layers(x)
        return logits

class FullyConnected3(FullyConnected):
    def __init__(self, *args, **kwargs):
        super(FullyConnected3, self).__init__(*args, **kwargs)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(self.labels)),
        )
    