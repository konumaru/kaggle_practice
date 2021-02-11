import copy
import numpy as np
import pandas as pd
from typing import List, Dict


class BaseTrainer(object):
    def __init__(self):
        self.model = None

    def fit(
        self,
        params: Dict,
        train_params: Dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
    ):
        NotImplementedError

    def predict(self, data):
        NotImplementedError

    def get_importance(self):
        NotImplementedError

    def get_model(self):
        NotImplementedError

    def set_seed(self, seed):
        NotImplementedError


class SklearnRegressionTrainer(BaseTrainer):
    def __init__(self, model):
        self.model = model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        weight: pd.DataFrame = None,
    ):
        self.model.fit(X, y, sample_weight=weight)
        self.feature_names_ = X.columns

    def predict(self, data):
        return self.model.predict(data)

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        importance = self.model.ffeature_importances_
        feature_name = self.feature_names_
        return dict(zip(feature_name, importance))

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
    ):
        self.model.fit(X, y, sample_weight=weight)
        self.feature_names_ = X.columns

    def predict(self, data):
        return self.model.predict_proba(data)[:, 1]

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        importance = self.model.ffeature_importances_
        feature_name = self.feature_names_
        return dict(zip(feature_name, importance))

    def get_model(self):
        return self.model
