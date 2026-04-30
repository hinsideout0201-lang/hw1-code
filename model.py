import numpy as np
from layers import Linear, ReLU, Sigmoid

class MLP:
    def __init__(self,in_dim,hidden_dim,num_classes,activate='ReLU'):
        if activate == 'ReLU':
            self.layers = [Linear(in_dim,hidden_dim,'ReLU'), ReLU(),
                           Linear(hidden_dim,hidden_dim,'ReLU'),
                           ReLU(),
                           Linear(hidden_dim,num_classes,'ReLU')]
        elif activate == 'Sigmoid':
            self.layers = [Linear(in_dim,hidden_dim,'Sigmoid'),
                           Sigmoid(),Linear(hidden_dim,hidden_dim,'Sigmoid'),
                           Sigmoid(), Linear(hidden_dim, num_classes,'Sigmoid')]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, out_grad):
        for layer in reversed(self.layers):
            out_grad = layer.backward(out_grad)
