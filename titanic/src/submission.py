import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd
import xgboost as xgb

from mikasa.common import timer
from mikasa.io import load_pickle
from mikasa.preprocessing import add_dummies


def submission(data, lr_models, rf_models, lgbm_models, xgb_models):
    def lgbm_predict(models, data):
        preds = [m.predict(data, num_iteration=m.best_iteration) for m in models]
        return np.mean(preds, axis=0)

    def xgb_predict(models, data):
        preds = [
            m.predict(xgb.DMatrix(data), ntree_limit=m.best_ntree_limit) for m in models
        ]
        return np.mean(preds, axis=0)

    def sklearn_predict(models, data):
        preds = [m.predict_proba(data)[:, 1] for m in models]
        return np.mean(preds, axis=0)

    pred = (
        0.0 * sklearn_predict(lr_models, data)
        + 0.1 * sklearn_predict(rf_models, data)
        + 0.3 * lgbm_predict(lgbm_models, data)
        + 0.6 * xgb_predict(xgb_models, data)
    )
    pred = (pred > 0.5).astype(np.int8)
    return pred


def raw_feature():
    data = pd.read_csv("../data/raw/test.csv")
    # Label encoding.
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    # One-hot encoding
    data = add_dummies(data, "Pclass")
    data["Embarked"].fillna("missing", inplace=True)
    data = add_dummies(data, "Embarked")
    # Fill null with average.
    data["Age"].fillna(30, inplace=True)
    data["Fare"].fillna(33, inplace=True)
    #
    # data["Name_length"] = data["Name"].apply(lambda x: len(x.split()))
    # data["Has_Cabin"] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    data["Fare_per_person"] = data.Fare / np.mean(data.SibSp + data.Parch + 1)

    data["Embarked_missing"] = 0
    use_cols = ["Sex", "SibSp", "Parch", "Age", "Fare", "Fare_per_person"]
    use_cols += list(data.columns[data.columns.str.contains("Pclass")])
    use_cols += list(data.columns[data.columns.str.contains("Embarked")])
    return data[use_cols]


def create_features():
    data = []
    data.append(raw_feature())

    data = pd.concat(data, axis=1)
    return data


def main():
    X_test = create_features()
    print(X_test.head())

    lr_models = load_pickle("../data/working/lr_models.pkl")
    rf_models = load_pickle("../data/working/rf_models.pkl")
    lgbm_models = load_pickle("../data/working/lgbm_models.pkl")
    xgb_models = load_pickle("../data/working/xgb_models.pkl")

    with timer("Submission"):
        pred = submission(X_test, lr_models, rf_models, lgbm_models, xgb_models)

        submit = pd.read_csv("../data/raw/gender_submission.csv")
        submit["Survived"] = pred
        submit.to_csv("../data/submit/submission.csv", index=False)

        print(submit.head())
        print(submit["Survived"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
