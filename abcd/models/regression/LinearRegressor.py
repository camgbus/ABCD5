"""PyTorch linear regressor
"""

from torch import nn
from abcd.models.regression.Regressor import Regressor

class LinearRegressor(Regressor):
    def __init__(self, *args, input_size=28, **kwargs):
        super(LinearRegressor, self).__init__(*args, **kwargs)
        self.input_size = input_size
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        output = self.linear(x)
        return output.float()