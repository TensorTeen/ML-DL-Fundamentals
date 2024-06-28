import numpy as np


class Utils:
    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_BCE(self, X, Y):
        return -np.sum(Y * np.log(X) + (1 - Y) * np.log(1 - X))

    def cost_MSE(self, X, Y):
        return np.sum((X - Y) ** 2)
