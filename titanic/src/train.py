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


def run_train(Trainer, params, X, y):
    trainer = Trainer()
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    cv_trainer = CrossValidationTrainer(cv, trainer)
    cv_trainer.fit(
        params=params["params"],
        train_params=params["train_params"],
        X=X,
        y=y,
    )

    models = cv_trainer.get_models()
    oof = (cv_trainer.get_oof() > 0.5).astype(int)
    return models, oof


def main():
    # TODO: Parameterをhydraで管理する
    lgbm_params = {
        "params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 300,
            "learning_rate": 0.1,
            "random_seed": 42,
            "max_depth": 2,
            "random_seed": 42,
            "verbose": -1,
        },
        "train_params": {
            "verbose_eval": 10,
            "num_boost_round": 1000,
            "early_stopping_rounds": 10,
        },
    }
    xgb_params = {
        "params": {
            "objective": "binary:logistic",
            "metric": "logloss",
            "learning_rate": 0.1,
            "random_seed": 42,
            "max_depth": 5,
            "gammma": 0.1,
            "colsample_bytree": 1,
            "min_child_weight": 1,
            "seed": 42,
            "verbose": -1,
        },
        "train_params": {
            "verbose_eval": 10,
            "num_boost_round": 500,
            "early_stopping_rounds": 10,
        },
    }
    stack_lgbm_params = {
        "params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 100,
            "learning_rate": 0.1,
            "random_seed": 42,
            "max_depth": 2,
            "random_seed": 42,
            "verbose": -1,
        },
        "train_params": {
            "verbose_eval": 10,
            "num_boost_round": 500,
            "early_stopping_rounds": 10,
        },
    }
    # Load Data
    feature_filepath = [
        "../data/feature/raw_feature.pkl",
    ]
    data = []
    for filepath in feature_filepath:
        feature = load_pickle(filepath)
        data.append(feature)

    X = pd.concat(data)
    y = load_pickle("../data/feature/target.pkl")

    # Train and Evaluation.
    with timer("train"):
        lgbm_models, lgbm_oof = run_train(LGBMTrainer, lgbm_params, X, y)
        xgb_models, xgb_oof = run_train(XGBTrainer, xgb_params, X, y)

    lgbm_metric = accuracy_score(y, (lgbm_oof > 0.5))
    print(f"AUC of LightGBM is {lgbm_metric:.8f}")

    xgb_metric = accuracy_score(y, (xgb_oof > 0.5))
    print(f"AUC of XGBoost is {xgb_metric:.8f}")

    ensemble_oof = np.mean([lgbm_oof, xgb_oof], axis=0)
    ensemble_oof = accuracy_score(y, (ensemble_oof > 0.5))
    print(f"AUC of Ensemble is {ensemble_oof:.8f}")

    # Stacking
    X_pred = pd.DataFrame({"lgbm": lgbm_oof, "xgb": xgb_oof})
    stack_models, stacked_oof = run_train(LGBMTrainer, stack_lgbm_params, X_pred, y)
    stacked_metric = accuracy_score(y, (stacked_oof > 0.5))
    print(f"AUC of Stacked LightGBM is {stacked_metric:.8f}")
    print("")

    # Dump models.
    dump_pickle(lgbm_models, "../data/working/lgbm_models.pkl")
    dump_pickle(xgb_models, "../data/working/xgb_models.pkl")
    dump_pickle(stack_models, "../data/working/stack_models.pkl")
    print("")


if __name__ == "__main__":
    main()
