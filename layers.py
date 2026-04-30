import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim, activate='ReLU'):
        if activate == 'ReLU':
            self.W = np.random.randn(in_dim,out_dim) * np.sqrt(2.0/in_dim)
        elif activate == 'Sigmoid':
            self.W = np.random.randn(in_dim,out_dim) * np.sqrt(1.0/in_dim)
        self.b = np.zeros((1,out_dim))
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        return x@self.W + self.b

    def backward(self, out_grad):
        self.dW = self.x.T@out_grad
        self.db = np.sum(out_grad, axis=0, keepdims=True)
        in_grad = out_grad@self.W.T
        return in_grad

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, out_grad):
        in_grad = out_grad * (self.x>0)
        return in_grad

class Sigmoid:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return 1 / (1+np.exp(-x))

    def backward(self, out_grad):
        s = 1 / (1+np.exp(-self.x))
        in_grad = out_grad * s * (1-s)
        return in_grad
