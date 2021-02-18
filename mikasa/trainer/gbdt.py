from typing import List, Dict

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost

from .base import BaseTrainer


class XGBTrainer(BaseTrainer):
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
        random_state: int = None,
    ):
        self.params["seed"] = random_state
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

    def get_model(self):
        return self.model

    def get_importance(self) -> Dict:
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        return self.model.get_score(importance_type="gain")


class LGBMTrainer(BaseTrainer):
    def __init__(self, params, train_params):
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
        random_state: int = None,
    ):
        self.params["seed"] = random_state
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

    def get_model(self):
        return self.model

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


class CatBoostTrainer(BaseTrainer):
    def __init__(self, params, train_params):
        self.model = catboost.CatBoost(params=params)
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
        random_state: int = None,
    ):
        self.model.set_params(random_seed=random_state)
        train_pool = catboost.Pool(
            X_train,
            label=y_train,
            weight=weight_train,
            cat_features=categorical_feature,
        )
        valid_pool = catboost.Pool(
            X_valid,
            label=y_valid,
            weight=weight_valid,
            cat_features=categorical_feature,
        )

        self.model.fit(
            X=train_pool,
            eval_set=valid_pool,
            **self.train_params,
        )

    def predict(self, data):
        pred = self.model.predict(data)
        return pred

    def get_model(self):
        return self.model

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        importance = self.model.feature_importances_
        feature_name = self.model.feature_names_
        return dict(zip(feature_name, importance))
