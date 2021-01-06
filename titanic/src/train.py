import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from mikasa.common import timer
from mikasa.io import load_pickle, dump_pickle
from mikasa.trainer.base import CrossValidationTrainer
from mikasa.trainer.gbdt import LGBMTrainer, XGBTrainer


def run_lgbm_train(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    trainer = LGBMTrainer()
    cv_trainer = CrossValidationTrainer(cv, trainer)
    cv_trainer.fit(
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 300,
            "learning_rate": 0.1,
            "random_seed": 42,
            "max_depth": 2,
            "verbose": -1,
        },
        train_params={
            "verbose_eval": 10,
            "num_boost_round": 1000,
            "early_stopping_rounds": 10,
        },
        X=X,
        y=y,
    )

    models = cv_trainer.get_models()
    oof = (cv_trainer.get_oof() > 0.5).astype(int)

    metric = accuracy_score(y, oof)
    return models, metric


def run_xgb_train(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    trainer = XGBTrainer()
    cv_trainer = CrossValidationTrainer(cv, trainer)
    cv_trainer.fit(
        params={
            "objective": "binary:logistic",
            "metric": "error",
            "learning_rate": 0.1,
            "random_seed": 42,
            "max_depth": 5,
            "gammma": 0.1,
            "colsample_bytree": 1,
            "min_child_weight": 1,
            "verbose": -1,
        },
        train_params={
            "verbose_eval": 10,
            "num_boost_round": 500,
            "early_stopping_rounds": 10,
        },
        X=X,
        y=y,
    )

    models = cv_trainer.get_models()
    oof = (cv_trainer.get_oof() > 0.5).astype(int)

    metric = accuracy_score(y, oof)
    return models, metric


def main():
    feature_filepath = [
        "../data/feature/raw_feature.pkl",
    ]
    data = []
    for filepath in feature_filepath:
        feature = load_pickle(filepath)
        data.append(feature)

    X = pd.concat(data)
    y = load_pickle("../data/feature/target.pkl")

    with timer("train"):
        lgbm_models, lgbm_metric = run_lgbm_train(X, y)
        xgb_models, xgb_metric = run_xgb_train(X, y)

    dump_pickle(lgbm_models, "../data/working/lgbm_models.pkl")
    dump_pickle(xgb_models, "../data/working/xgb_models.pkl")

    print(f"AUC of LightGBM is {lgbm_metric:.8f}")
    print(f"AUC of XGBoost is {xgb_metric:.8f}")


if __name__ == "__main__":
    main()
