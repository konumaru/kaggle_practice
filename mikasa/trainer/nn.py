from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from base import BaseTrainer


class TabNetClassificationTrainer(BaseTrainer):
    def __init__(self):
        """[summary]

        Parameters
        ----------
        objective : str, optional
            objective is only regression or classification.
        """
        self.model = None
        self.feature_names = None

    def fit(
        self,
        params: Dict,
        train_params: Dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        categorical_feature: List[str] = None,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
    ):

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy().ravel()
        X_valid = X_valid.to_numpy()
        y_valid = y_valid.to_numpy().ravel()

        self.model = TabNetClassifier(**params)
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=["train", "valid"],
            **train_params,
        )

    def predict(self, data):
        pred = self.model.predict_proba(data.to_numpy())[:, 1]
        return pred

    def get_importance(self):
        NotImplementedError

    def get_model(self):
        return self.model


class TabNetRegressionTrainer(BaseTrainer):
    def __init__(self):
        """[summary]

        Parameters
        ----------
        objective : str, optional
            objective is only regression or classification.
        """
        self.model = None
        self.feature_names = None

    def fit(
        self,
        params: Dict,
        train_params: Dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        categorical_feature: List[str] = None,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
    ):

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_valid = X_valid.to_numpy()
        y_valid = y_valid.to_numpy()

        self.model = TabNetRegressor(**params)
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=["train", "valid"],
            **train_params,
        )

    def predict(self, data):
        pred = self.model.predict(data.to_numpy())
        return pred

    def get_importance(self):
        NotImplementedError

    def get_model(self):
        return self.model


def train_model(X, Y, Trainer, metric="auc", metric_func=metrics.roc_auc_score):
    X, X_test, Y, y_test = train_test_split(X, Y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y)

    X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
    X_valid, y_valid = pd.DataFrame(X_valid), pd.DataFrame(y_valid)
    X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)

    trainer = Trainer()
    trainer.fit(
        {
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=2e-2),
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "scheduler_params": {
                "step_size": 10,
                "gamma": 0.9,
            },
            "device_name": "auto",
        },
        {
            "eval_metric": [metric],
            "max_epochs": 10,
            "patience": 5,
            "batch_size": 1024,
            "num_workers": 0,
            "drop_last": False,
        },
        X_train,
        y_train,
        X_valid,
        y_valid,
        y_train,
    )

    pred = trainer.predict(X_test)
    score = metric_func(y_test, pred)
    print(f"{metric} is", score)


if __name__ == "__main__":
    X, Y = make_regression(
        random_state=12,
        n_samples=10_000,
        n_features=100,
        n_informative=20,
        n_targets=1,
    )
    Y = Y.reshape(-1, 1)

    train_model(
        X,
        Y,
        TabNetRegressionTrainer,
        metric="mae",
        metric_func=metrics.mean_absolute_error,
    )

    X, Y = make_classification(
        random_state=12,
        n_samples=10_000,
        n_features=100,
        n_redundant=3,
        n_informative=20,
        n_clusters_per_class=1,
        n_classes=2,
    )
    train_model(X, Y, TabNetClassificationTrainer, metric="auc")
