import numpy as np
from operator import truediv, add, sub
from .utils import Utils


class Optimizers:
    def __init__(self, FFN, loss_fn):
        self.FFN = FFN
        self.loss = []
        self.utils = Utils()
        self.loss_fn = loss_fn

    def gradient_descent(self, num_epochs, X_train, Y_train, eta, print_loss=False):
        for _ in range(num_epochs):
            n = len(X_train)
            dW = [
                np.zeros([self.FFN.N[i], self.FFN.N[i + 1]])
                for i in range(self.FFN.L - 1)
            ]
            db = [np.zeros(([1, self.FFN.N[i + 1]])) for i in range(self.FFN.L - 1)]
            # print([db[i].shape for i in range(len(db))])
            for i in range(n):
                dWi, dbi = self.back_prop(X_train[i], Y_train[i])
                dWi = list(map(truediv, dWi, [n * (1 / eta) for _ in range(len(dWi))]))
                dbi = list(map(truediv, dbi, [n * (1 / eta) for _ in range(len(dbi))]))
                dW = list(map(add, dW, dWi))
                db = list(map(add, db, dbi))

            self.FFN.W = list(map(sub, self.FFN.W, dW))
            self.FFN.b = list(map(sub, self.FFN.b, db))
            Y_pred = self.FFN(X_train[0])
            if print_loss:
                print(f"Loss: {self.loss_fn(Y_pred,Y_train[0])}")
            self.loss.append(self.loss_fn(Y_pred, Y_train[0]))

    def gradient_descent_with_momentum(
        self, num_epochs, X_train, Y_train, eta, beta, print_loss=False
    ):
        for _ in range(num_epochs):
            n = len(X_train)
            dW = [
                np.zeros([self.FFN.N[i], self.FFN.N[i + 1]])
                for i in range(self.FFN.L - 1)
            ]
            db = [np.zeros(([1, self.FFN.N[i + 1]])) for i in range(self.FFN.L - 1)]
            duW = [
                np.zeros([self.FFN.N[i], self.FFN.N[i + 1]])
                for i in range(self.FFN.L - 1)
            ]
            dub = [np.zeros(([1, self.FFN.N[i + 1]])) for i in range(self.FFN.L - 1)]
            # print([db[i].shape for i in range(len(db))])
            for i in range(n):
                dWi, dbi = self.back_prop(X_train[i], Y_train[i])
                dWi = list(map(truediv, dWi, [n * (1 / eta) for _ in range(len(dWi))]))
                dbi = list(map(truediv, dbi, [n * (1 / eta) for _ in range(len(dbi))]))
                dW = list(map(add, dW, dWi))
                db = list(map(add, db, dbi))
            duW = [beta * i for i in duW]
            duW = list(map(add, duW, dW))
            dub = list(map(add, dub, db))
            self.FFN.W = list(map(sub, self.FFN.W, duW))
            self.FFN.b = list(map(sub, self.FFN.b, dub))
            Y_pred = self.FFN(X_train[0])
            if print_loss:
                print(f"Loss: {self.loss_fn(Y_pred,Y_train[0])}")
            self.loss.append(self.loss_fn(Y_pred, Y_train[0]))

    def back_prop(self, X, Y):
        A, H = self.FFN.forward_prop(X)
        dW = [None] * (self.FFN.L - 1)
        db = [None] * (self.FFN.L - 1)
        dA = [None] * (self.FFN.L)
        dA[self.FFN.L - 1] = (
            (Y / H[-1] - (1 - Y) / (1 - H[-1])) * H[-1] * (1 - H[-1]).reshape(1, 1)
        )
        dH = [None] * (self.FFN.L - 1)
        for k in range(self.FFN.L - 1, 0, -1):
            dWk = dA[k].T @ H[k - 1]
            dW[k - 1] = dWk.T
            db[k - 1] = dA[k]
            dH[k - 1] = dA[k] @ self.FFN.W[k - 1].T
            dA[k - 1] = dH[k - 1] * H[k - 1] * (1 - H[k - 1])
        return dW, db
