"""Classifiers made excludively out of fully-connected layers.
"""

from torch import nn
from abcd.models.classification.Classifier import Classifier

class FullyConnected(Classifier):
    def __init__(self, *args, **kwargs):
        super(FullyConnected, self).__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_layers(x)
        return logits

class FullyConnected3(FullyConnected):
    def __init__(self, *args, **kwargs):
        super(FullyConnected3, self).__init__(*args, **kwargs)
        self.linear_layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    