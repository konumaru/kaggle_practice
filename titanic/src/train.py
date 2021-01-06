import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from mikasa.common import timer
from mikasa.io import load_pickle
from mikasa.trainer.gbdt import LGBMTrainer


def run_train(X, y):
    models = []
    oof = np.zeros(y.shape[0])
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        trainer = LGBMTrainer()
        trainer.fit(
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
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            # categorical_feature=["Pclass"],
        )

        models.append(trainer.get_model())
        oof[valid_idx] = (trainer.predict(X_valid) > 0.5).astype(int)

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
        models, metric = run_train(X, y)

    print(metric)


if __name__ == "__main__":
    main()
