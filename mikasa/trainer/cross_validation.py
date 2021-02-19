import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from .base import BaseTrainer
from ..common import seed_everything


class CrossValidationTrainer(BaseTrainer):
    """Trainer of Cross Validation.

    Example
    -------
    def run_train(Trainer, params, X, y):
        trainer = Trainer()
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        cv_trainer = CrossValidationTrainer(cv, trainer)
        cv_trainer.fit(
            params=params,
            train_params=train_params,
            X=X,
            y=y,
        )

        models = cv_trainer.get_models()
        oof = (cv_trainer.get_oof() > 0.5).astype(int)
        return models, oof

    models, oof = run_train(LGBMTrainer, params, train_params, X, y)
    """

    def __init__(self, cv, trainer):
        self.cv = cv
        self.trainer = trainer
        self.trainers = list()
        self.oof = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        weight: np.ndarray = None,
        categorical_feature: List[str] = None,
        random_state: int = None,
        groups: np.ndarray = None,
    ):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.oof = np.full_like(np.zeros(y.shape[0]), np.nan, dtype=np.double)

        for n_fold, (train_idx, valid_idx) in enumerate(
            self.cv.split(X, y, groups=groups)
        ):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

            if weight is not None:
                weight_train, weight_valid = weight[train_idx], weight[valid_idx]
            else:
                weight_train, weight_valid = None, None

            _trainer = copy.deepcopy(self.trainer)
            if "Sklearn" in _trainer.__class__.__name__:
                _trainer.fit(
                    X_train,
                    y_train,
                    weight=weight_train,
                    random_state=random_state,
                )
            else:
                _trainer.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    weight_train=weight_train,
                    weight_valid=weight_valid,
                    categorical_feature=categorical_feature,
                    random_state=random_state,
                )

            self.trainers.append(_trainer)
            self.oof[valid_idx] = _trainer.predict(X_valid)

    def predict(self, data) -> List:
        preds = [t.predict(data) for t in self.trainers]
        return preds

    def get_model(self) -> List:
        models = [t.get_model() for t in self.trainers]
        return models

    def get_importance(
        self, max_feature: int = 50
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        if hasattr(self.trainers[0], "get_importance"):
            importances = [t.get_importance() for t in self.trainers]
            importances = pd.DataFrame(importances).T

            importances = importances.assign(
                mean_feature_importance=importances.mean(axis=1),
                std_feature_importance=importances.std(axis=1),
            )
            importances = importances.sort_values(by="mean_feature_importance")

            if max_feature is not None:
                importances = importances.iloc[:max_feature]

            name = importances.index.tolist()
            mean_importance = importances["mean_feature_importance"].to_numpy()
            std_importance = importances["std_feature_importance"].to_numpy()
            return (name, mean_importance, std_importance)
        else:
            name = []
            mean_importance = np.array([])
            std_importance = np.array([])
            return (name, mean_importance, std_importance)

    def get_oof(self):
        return self.oof


class RSACVTrainer(BaseTrainer):
    """Trainer of random seed averaging cross validation."""

    def __init__(self, cv, trainer, seed=42):
        self.cv = cv
        self.trainer = trainer
        self.seed = seed
        self.trainers = list()
        self.oof = None

    def fit(
        self,
        X,
        y,
        categorical_feature: List[str] = None,
        weight: np.ndarray = None,
        groups=None,
        num_seed=5,
    ):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.oof = np.full_like(np.zeros(y.shape[0]), np.nan, dtype=np.double)
        for i in range(num_seed):
            seed = self.seed + i
            for n_fold, (train_idx, valid_idx) in enumerate(
                self.cv.split(X, y, groups=groups)
            ):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

                if weight is not None:
                    weight_train, weight_valid = weight[train_idx], weight[valid_idx]
                else:
                    weight_train, weight_valid = None, None

                if "Sklearn" in self.trainer.__class__.__name__:
                    _trainer = copy.deepcopy(self.trainer)
                    _trainer.fit(X_train, y_train)
                else:
                    _trainer = copy.deepcopy(self.trainer)
                    _trainer.set_seed(seed)
                    _trainer.fit(
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        weight_train=weight_train,
                        weight_valid=weight_valid,
                        categorical_feature=categorical_feature,
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
