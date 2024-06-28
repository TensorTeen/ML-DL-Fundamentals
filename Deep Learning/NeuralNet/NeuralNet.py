import numpy as np
from .utils import Utils


class NeuralNet:
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.W = [np.random.randn(N[i], N[i + 1]) for i in range(L - 1)]
        self.b = [np.random.randn(1, N[i + 1]) for i in range(L - 1)]
        self.loss = []
        self.utils = Utils()

    def forward_prop(self, X):
        H = [X]
        A = []
        for i in range(self.L - 1):
            a_i = H[i] @ self.W[i] + self.b[i]
            h_i = self.utils.sigmoid(a_i)
            A.append(a_i)
            H.append(h_i)
        return A, H

    def back_prop(self, X, Y):
        A, H = self.forward_prop(X)
        dW = [None] * (self.L - 1)
        db = [None] * (self.L - 1)
        dA = [None] * (self.L)
        dA[self.L - 1] = (
            (Y / H[-1] - (1 - Y) / (1 - H[-1])) * H[-1] * (1 - H[-1]).reshape(1, 1)
        )
        dH = [None] * (self.L - 1)
        for k in range(self.L - 1, 0, -1):
            dWk = dA[k].T @ H[k - 1]
            dW[k - 1] = dWk.T
            db[k - 1] = dA[k]
            dH[k - 1] = dA[k] @ self.W[k - 1].T
            dA[k - 1] = dH[k - 1] * H[k - 1] * (1 - H[k - 1])
        return dW, db

    def __call__(self, X):
        A, H = self.forward_prop(X)
        return H[-1]
