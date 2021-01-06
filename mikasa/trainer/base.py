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


class CrossValidationTrainer(object):
    def __init__(self, cv, trainer):
        self.cv = cv
        self.trainer = trainer
        self.trainers = list()
        self.oof = None

    def fit(self, params, train_params, X, y, weight=None, groups=None):
        self.oof = np.zeros(y.shape[0])
        for n_fold, (train_idx, valid_idx) in enumerate(
            self.cv.split(X, y, groups=groups)
        ):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

            _trainer = copy.deepcopy(self.trainer)
            _trainer.fit(
                params=params,
                train_params=train_params,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            self.trainers.append(_trainer)
            self.oof[valid_idx] = _trainer.predict(X_valid)

    def predict(self, data):
        preds = [t.predict(data) for t in self.trainers]
        return preds

    def get_importance(self, max_feature: int = 50):
        importances = [t.get_importance() for t in self.trainers]
        importances = pd.DataFrame(importances).T

        importances = importances.assign(
            mean_feature_importance=importances.mean(axis=1),
            std_feature_importance=importances.std(axis=1),
        )
        importances = importances.sort_values(by="mean_feature_importance")

        if max_feature is not None:
            importances = importances.iloc[:max_feature]

        name = importances.index
        mean_importance = importances["mean_feature_importance"]
        std_importance = importances["std_feature_importance"]
        return (name, mean_importance, std_importance)

    def get_models(self):
        models = [t.get_model() for t in self.trainers]
        return models

    def get_oof(self):
        return self.oof
