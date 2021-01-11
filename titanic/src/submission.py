import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd
import xgboost as xgb

from mikasa.common import timer
from mikasa.io import load_pickle


def submission(data, lgbm_models, xgb_models, stack_models):
    def lgbm_predict(models, data):
        preds = [m.predict(data, num_iteration=m.best_iteration) for m in models]
        return np.mean(preds, axis=0)

    def xgb_predict(models, data):
        preds = [
            m.predict(xgb.DMatrix(data), ntree_limit=m.best_ntree_limit) for m in models
        ]
        return np.mean(preds, axis=0)

    lgbm_pred = lgbm_predict(lgbm_models, data)
    xgb_pred = xgb_predict(xgb_models, data)

    # NOTE: 現状のStackモデルでは過学習している。
    # pred = pd.DataFrame({"lgbm": lgbm_pred, "xgb": xgb_pred})
    # pred = lgbm_predict(stack_models, pred)

    pred = 0.5 * lgbm_pred + 0.5 * xgb_pred
    pred = (pred > 0.5).astype(np.int8)
    return pred


def create_features(data):
    # === extract_raw_feature ===
    # Fill null values.
    # [
    #     "Pclass",
    #     "Sex",
    #     "Age",
    #     "SibSp",
    #     "Parch",
    #     "Ticket",
    #     "Fare",
    #     "Cabin",
    #     "Embarked",
    #     "family_size",
    #     "is_group_guest",
    # ]
    data["Embarked"].fillna("missing", inplace=True)

    # Label encoding.
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2, "missing": 3})
    ticket_uniques = load_pickle("../data/preprocess/ticket_uniques.pkl", verbose=False)
    data["Ticket"] = data["Ticket"].map({u: i for i, u in enumerate(ticket_uniques)})
    cabin_uniques = load_pickle("../data/preprocess/cabin_uniques.pkl", verbose=False)
    data["Cabin"] = data["Cabin"].map({u: i for i, u in enumerate(cabin_uniques)})

    # Fill null Age by honorific
    honorific_avgAge = [("Mr.", 32), ("Miss.", 21), ("Mrs.", 37), ("Mr.", 5)]
    for t, val in honorific_avgAge:
        t_idx = data["Name"].str.contains(t)
        data["Age"][t_idx].fillna(val, inplace=True)
    honorific_avgFare = [("Mr.", 24), ("Miss.", 42), ("Mrs.", 49), ("Mr.", 36)]
    for t, val in honorific_avgAge:
        t_idx = data["Name"].str.contains(t)
        data["Fare"][t_idx].fillna(val, inplace=True)
    # Honorific features
    data["family_size"] = data["SibSp"] + data["Parch"] + 1
    data["is_group_guest"] = np.where(data["family_size"] >= 1, 1, 0)

    # Drop columns
    data.drop(["Survived", "PassengerId", "Name"], axis=1, inplace=True)
    print(data.columns)
    return data


def main():
    test = pd.read_csv("../data/raw/test.csv")
    test["Survived"] = 0
    X_test = create_features(test)

    lgbm_models = load_pickle("../data/working/lgbm_models.pkl")
    xgb_models = load_pickle("../data/working/xgb_models.pkl")
    stack_models = load_pickle("../data/working/stack_models.pkl")

    print(test.head())

    with timer("Submission"):
        pred = submission(X_test, lgbm_models, xgb_models, stack_models)

        submit = pd.read_csv("../data/raw/gender_submission.csv")
        submit["Survived"] = pred
        submit.to_csv("../data/submit/submission.csv", index=False)

        print(submit.head())
        print(submit["Survived"].value_counts().sort_index())


if __name__ == "__main__":
    main()
