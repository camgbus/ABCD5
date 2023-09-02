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

        self.sigmoid = nn.Sigmoid() #output range is (0, 1)
    
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
        

"MLP Regressor with customizable architecture based on parameters"
class MLPRegressorCustom(MLPRegressor):
    """
    Series of linear layers followed by ReLU activations, with the final layer having a sigmoid activation.
    """
    def __init__(self, hidden_sizes, dropout_p=0.0, *args, **kwargs):
        super(MLPRegressorCustom, self).__init__(*args, **kwargs)
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes) + 1 #+1 for final layer
        self.dropout_p = dropout_p

        #create layers
        layers = []
        curr_in_size = self.input_size
        for i in range(self.num_layers-1): # linear -> ReLU -> dropout
            curr_out_size = self.hidden_sizes[i]
            layers.append(nn.Linear(curr_in_size, curr_out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_p))
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
        
        self.linear_layers = self.linear1