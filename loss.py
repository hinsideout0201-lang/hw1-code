import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.p = None
        self.y = None

    def forward(self, x, y):
        x = x - np.max(x,axis=1,keepdims=True)
        ex = np.exp(x)
        p = ex / np.sum(ex,axis=1,keepdims=True)

        self.p = p
        self.y = y

        n = x.shape[0]
        loss = -np.log(p[np.arange(n), y]).mean()
        return loss

    def backward(self):
        n = self.p.shape[0]
        grad = self.p.copy()
        grad[np.arange(n), self.y] -= 1
        grad /= n
        return grad
