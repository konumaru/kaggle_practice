import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from mikasa.common import timer
from mikasa.trainer.gbdt import LGBMTrainer


def preprocess(data):
    drop_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
    data.drop(drop_cols, axis=1, inplace=True)

    target = "Survived"
    features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    return data, target, features


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


def submission(models, data):
    def lgbm_predict(models, data):
        preds = [m.predict(data) for m in models]
        return np.mean(preds, axis=0)

    pred = lgbm_predict(models, data)
    pred = (pred > 0.5).astype(np.int8)
    return pred


def main():
    with timer("Load and Preprocess"):
        train = pd.read_csv("../data/raw/train.csv")
        train, target, features = preprocess(train)

        test = pd.read_csv("../data/raw/test.csv")
        test, _, _ = preprocess(test)

    print(train.head())

    with timer("train"):
        X = train[features]
        y = train[target]
        models, metric = run_train(X, y)

    print(metric)

    with timer("Submission"):
        pred = submission(models, test[features])

    test = pd.read_csv("../data/raw/test.csv")
    submit = test[["PassengerId"]].copy()
    submit["Survived"] = pred
    submit.to_csv("../data/submit/submission.csv", index=False)
    print(submit.head())


if __name__ == "__main__":
    main()
