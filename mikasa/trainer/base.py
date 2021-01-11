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


class CrossValidationTrainer(object):
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

    def fit(self, X, y, weight: np.ndarray = None, groups=None):
        self.oof = np.zeros(y.shape[0])
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
                _trainer.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    weight_train=weight_train,
                    weight_valid=weight_valid,
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
