from typing import List, Dict

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

from .base import BaseTrainer


class XGBTrainer(BaseTrainer):
    def __init__(self, params, train_params):
        self.model = None
        self.params = params
        self.train_params = train_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
    ):
        train_dataset = xgb.DMatrix(X_train, label=y_train, weight=weight_train)
        valid_dataset = xgb.DMatrix(X_valid, label=y_valid, weight=weight_valid)

        self.model = xgb.train(
            self.params,
            train_dataset,
            evals=[(train_dataset, "train"), (valid_dataset, "valid")],
            **self.train_params,
        )

    def predict(self, data):
        pred = self.model.predict(
            xgb.DMatrix(data), ntree_limit=self.model.best_ntree_limit
        )
        return pred

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        return self.model.get_score(importance_type="gain")

    def get_model(self):
        return self.model


class LGBMTrainer(BaseTrainer):
    def __init__(self, params, train_params):
        self.model = None
        self.params = params
        self.train_params = train_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        categorical_feature: List[str] = None,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
    ):
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=weight_train,
            categorical_feature=categorical_feature,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            weight=weight_valid,
        )

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, valid_data],
            **self.train_params,
        )

    def predict(self, data):
        pred = self.model.predict(data, num_iteration=self.model.best_iteration)
        return pred

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        importance = self.model.feature_importance(
            importance_type="gain", iteration=self.model.best_iteration
        )
        feature_name = self.model.feature_name()
        return dict(zip(feature_name, importance))

    def get_model(self):
        return self.model


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X, Y = make_classification(
        random_state=12,
        n_samples=10_000,
        n_features=100,
        n_redundant=3,
        n_informative=20,
        n_clusters_per_class=1,
        n_classes=2,
    )

    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=0.2, stratify=Y
    )

    trainer = LGBMTrainer()
    trainer.fit({"verbosity": -1}, {}, X_train, y_train, X_valid, y_valid)
    pred = trainer.predict(X_test)

    auc = roc_auc_score(y_test, pred)
    print("AUC is", auc)


if __name__ == "__main__":
    main()
