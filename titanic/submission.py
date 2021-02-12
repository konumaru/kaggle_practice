import os
import sys

sys.path.append("..")

import numpy as np
import pandas as pd


import config
from mikasa.common import timer
from mikasa.io import save_cache, load_pickle, dump_pickle, load_feature
from mikasa.preprocessing import add_dummies

dump_dir = "../data/titanic/test_feature/"


@save_cache(os.path.join(dump_dir, "raw_feature.pkl"), use_cache=False)
def raw_feature(data: pd.DataFrame):
    # Label encoding.
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    # One-hot encoding
    data = add_dummies(data, "Pclass")
    data["Embarked"].fillna("missing", inplace=True)
    data = add_dummies(data, "Embarked")
    # Fill null with average.
    data["Age"].fillna(30, inplace=True)
    data["Fare"].fillna(33, inplace=True)

    # data["Name_length"] = data["Name"].apply(lambda x: len(x.split()))
    # data["Has_Cabin"] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    data["Fare_per_person"] = data.Fare / np.mean(data.SibSp + data.Parch + 1)

    use_cols = ["Sex", "SibSp", "Parch", "Age", "Fare", "Fare_per_person"]
    use_cols += list(data.columns[data.columns.str.contains("Pclass")])
    use_cols += list(data.columns[data.columns.str.contains("Embarked")])
    return data[use_cols]


@save_cache(os.path.join(dump_dir, "cabin_feature.pkl"), use_cache=False)
def cabin_feature(data: pd.DataFrame):
    data["Cabin_head"] = data["Cabin"].str.extract(r"(\w)[\d+]", expand=False)
    data["Cabin_head"].fillna("H", inplace=True)

    cabin_head_map = {s: i for i, s in enumerate("ABCDEFGH")}
    data["Cabin_head"] = data["Cabin_head"].map(cabin_head_map)
    data = add_dummies(data, "Cabin_head")
    return data[list(data.columns[data.columns.str.contains("Cabin_head")])]


def create_features(data: pd.DataFrame):
    raw_feature(data.copy())
    cabin_feature(data.copy())


def lgbm_predict(models, data):
    preds = [
        m.predict(
            data, num_iteration=m.best_iteration, predict_disable_shape_check=True
        )
        for m in models
    ]
    return np.array(preds)


def sklearn_predict(models, data):
    preds = [m.predict(data) for m in models]
    return np.array(preds)


def main():
    # Create feature.
    test = pd.read_csv("../data/titanic/raw/test.csv")
    print(test.head())
    create_features(test)
    # Join all feature.
    feature_files = config.FeatureList.features
    feature_files = [
        os.path.join(dump_dir, f"{filename}.pkl") for filename in feature_files
    ]
    X = load_feature(feature_files)
    print(X.head())

    # First layer prediction.
    lgbm_models = load_pickle("../data/titanic/model/LGBM_models.pkl")

    # Prediction
    pred_first = []
    pred_first.append(lgbm_predict(lgbm_models, X).T)
    pred_first = np.concatenate(pred_first, axis=1)
    pred_first = pd.DataFrame(pred_first)
    print(pred_first.shape)
    print(pred_first.head())

    # Second layer prediction.
    stack_models = load_pickle("../data/titanic/model/stack_ridge_models.pkl")
    pred_second = sklearn_predict(stack_models, pred_first)
    pred_second = np.where(np.mean(pred_second, axis=0) > 0.5, 1, 0)
    print(pred_second)

    # Dump submission.csv
    submission = test[["PassengerId"]].copy()
    submission["Survived"] = pred_second
    submission.to_csv("../data/titanic/working/submission.csv", index=False)
    print(submission.shape)
    print(submission.head())


if __name__ == "__main__":
    main()
