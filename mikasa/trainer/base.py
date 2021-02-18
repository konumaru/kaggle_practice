from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self):
        NotImplementedError

    @abstractmethod
    def predict(self):
        NotImplementedError

    @abstractmethod
    def get_model(self):
        NotImplementedError


class SklearnRegressionTrainer(BaseTrainer):
    def __init__(self, model):
        self.model = model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        weight: pd.DataFrame = None,
        random_state: int = None,
    ):
        np.random.seed(random_state)
        self.model.fit(X, y, sample_weight=weight)

    def predict(self, data):
        return self.model.predict(data)

    def get_model(self):
        return self.model


class SklearnClassificationTrainer(BaseTrainer):
    def __init__(self, model):
        self.model = model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        weight: pd.DataFrame = None,
        random_state: int = None,
    ):
        np.random.seed(random_state)
        self.model.fit(X, y, sample_weight=weight)

    def predict(self, data):
        return self.model.predict_proba(data)[:, 1]

    def get_model(self):
        return self.model
