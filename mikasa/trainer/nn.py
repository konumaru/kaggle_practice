import numpy as np
import pandas as pd
from typing import List, Dict

from sklearn import metrics
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from .base import BaseTrainer


class TabNetClassificationTrainer(BaseTrainer):
    def __init__(self, params: Dict = {}, train_params: Dict = {}):
        self.model = None
        self.params = params
        self.train_params = train_params

    def fit(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.DataFrame,
        y_valid: pd.DataFrame,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
        categorical_feature: List[str] = None,
        random_state: int = 0,
    ):
        self.params["seed"] = random_state

        X_train = self._convert_dataframe_to_ndarray(X_train)
        y_train = self._convert_dataframe_to_ndarray(y_train)
        X_valid = self._convert_dataframe_to_ndarray(X_valid)
        y_valid = self._convert_dataframe_to_ndarray(y_valid)

        self.model = TabNetClassifier(**self.params)
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=["train", "valid"],
            **self.train_params,
        )

    def predict(self, data):
        data = self._convert_dataframe_to_ndarray(data)
        pred = self.model.predict_proba(data)[:, 1]
        return pred

    def get_model(self):
        return self.model

    def _convert_dataframe_to_ndarray(self, data: pd.DataFrame):
        if type(data) == pd.DataFrame:
            return data.to_numpy()
        else:
            return data


class TabNetRegressionTrainer(BaseTrainer):
    def __init__(self, params: Dict = {}, train_params: Dict = {}):
        self.model = None
        self.params = params
        self.train_params = train_params

    def fit(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.DataFrame,
        y_valid: pd.DataFrame,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
        categorical_feature: List[str] = None,
        random_state: int = 0,
    ):
        self.params["seed"] = random_state

        X_train = self._convert_dataframe_to_ndarray(X_train)
        y_train = self._convert_dataframe_to_ndarray(y_train)
        X_valid = self._convert_dataframe_to_ndarray(X_valid)
        y_valid = self._convert_dataframe_to_ndarray(y_valid)

        self.model = TabNetRegressor(**self.params)
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=["train", "valid"],
            **self.train_params,
        )

    def predict(self, data):
        data = self._convert_dataframe_to_ndarray(data)
        pred = self.model.predict(data)
        return pred

    def get_model(self):
        return self.model

    def _convert_dataframe_to_ndarray(self, data: pd.DataFrame):
        if type(data) == pd.DataFrame:
            return data.to_numpy()
        else:
            return data
