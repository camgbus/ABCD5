"""Pytorch regressors with only fully connected layers.
"""

import torch
from torch import nn
import torch.nn.init as init
from abcd.models.Model import Model

"General, parent MLP Regressor class"
class MLPRegressor(Model):
    def __init__(self, *args, input_size, output_size=1, **kwargs):
        super(MLPRegressor, self).__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size

        self.linear_layers = None #will be populated by children models

        self.sigmoid = nn.Sigmoid() #oupout range is (0, 1)
    
    def forward(self, x):
        '''Forward pass for MLP regression model'''
        x = self.linear_layers(x)
        x = self.sigmoid(x).squeeze()
        return x
    
    def predict(self, x):
        '''Make predictions based on given input'''
        self.eval() #set the model to evaluation mode
        with torch.no_grad():
            pred = self(x)
            return pred.detach()
        

"MLP Regressor which customizes architecture based on parameters"
class MLPRegressorCustom(MLPRegressor):
    """
    Series of linear layers followed by ReLU activations, with the final layer having a sigmoid activation.
    """
    def __init__(self, num_layers, hidden_sizes, *args, **kwargs):
        super(MLPRegressorCustom, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes

        assert self.num_layers == len(self.hidden_sizes) + 1

        #create layers
        layers = []
        curr_in_size = self.input_size
        for i in range(num_layers-1):
            curr_out_size = self.hidden_sizes[i]
            layers.append(nn.Linear(curr_in_size, curr_out_size))
            layers.append(nn.ReLU())
            curr_in_size = curr_out_size
        layers.append(nn.Linear(curr_in_size, self.output_size)) #final layer

        #xavier uniform weight initialization & zero biases
        for layer in layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        
        #set linear layers
        self.linear_layers = nn.Sequential(*layers)


"Linear Regressor"
class LinearRegressor(MLPRegressor):
    def __init__(self, *args, **kwargs):
        super(LinearRegressor, self).__init__(*args, **kwargs)

        self.linear1 = nn.Linear(self.input_size, 1)

        init.xavier_uniform_(self.linear1.weight) #xavier uniform weight initialization
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_() #initialize bias to 0
        
        #self.linear_layers = self.linear1
        self.linear_layers = self.linear1


"MLP Regressor with 3 Layers"
class MLPRegressor3(MLPRegressor):
    def __init__(self, hidden_sizes=[256, 64], *args, **kwargs):
        super(MLPRegressor3, self).__init__(*args, **kwargs)

        self.hidden_sizes = hidden_sizes

        # Three linear layers
        self.linear1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.linear2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.linear3 = nn.Linear(self.hidden_sizes[1], self.output_size)

        # Apply Xavier uniform initialization to the weights of the linear layers
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

        # Initialize biases to 0
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.linear3.bias is not None:
            self.linear3.bias.data.zero_()

        self.linear_layers = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3
        )