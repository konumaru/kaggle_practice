import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BaseEnsembler(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        NotImplementedError

    def predict(self, data):
        NotImplementedError


class SimpleAgerageEnsember(object):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        pass

    def predict(self, X: np.ndarray):
        return np.mean(X, axis=1)


class ManualWeightedEnsember(object):
    def __init__(self, weights):
        self.weights = weights

    def fit(self, X, y):
        assert X.shape[1] == len(self.weights), "Must Same dim of X and weights."

    def predict(self, data: np.ndarray):
        pred = np.zeros(data.shape[0])
        for i, weight in enumerate(self.weights):
            pred += weight * data[:, i]
        return pred


class VotingEnsember(object):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        NotImplementedError

    def predict(self, data):
        NotImplementedError
