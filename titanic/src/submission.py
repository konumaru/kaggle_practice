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
    # === family_feature ===
    data["family_size"] = data["SibSp"] + data["Parch"] + 1
    data["is_group_guest"] = np.where(data["family_size"] >= 1, 1, 0)
    # === ticket_type ===
    data["ticket_type"] = data["Ticket"].apply(lambda x: x[0:3])
    cabin_uniques = load_pickle(
        "../data/preprocess/ticketType_uniques.pkl", verbose=False
    )
    data["ticket_type"] = data["ticket_type"].map(
        {u: i for i, u in enumerate(cabin_uniques)}
    )
    # === fare_rank ===
    data["Fare_rank"] = -1
    data.loc[data["Fare"] <= 7.91, "Fare_rank"] = 0
    data.loc[(data["Fare"] > 7.91) & (data["Fare"] <= 14.454), "Fare_rank"] = 1
    data.loc[(data["Fare"] > 14.454) & (data["Fare"] <= 31), "Fare_rank"] = 2
    data.loc[data["Fare"] > 31, "Fare_rank"] = 3
    data["Fare_rank"] = data["Fare_rank"].astype(int)
    data["FareRank_Pclass"] = data["Fare_rank"] * data["Pclass"]
    # === age_rank ===
    data["Age_rank"] = -1
    data.loc[data["Age"] <= 16, "Age_rank"] = 0
    data.loc[(data["Age"] > 16) & (data["Age"] <= 32), "Age_rank"] = 1
    data.loc[(data["Age"] > 32) & (data["Age"] <= 48), "Age_rank"] = 2
    data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age_rank"] = 3
    data.loc[data["Age"] > 64, "Age_rank"] = 4
    data["Age_rank"] = data["Age_rank"].astype(int)
    data["AgeRank_Pclass"] = data["Age_rank"] * data["Pclass"]
    # === name_feature ===
    data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    data["Title"] = data["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )

    data["Title"] = data["Title"].replace("Mlle", "Miss")
    data["Title"] = data["Title"].replace("Ms", "Miss")
    data["Title"] = data["Title"].replace("Mme", "Mrs")
    # Fill null Age by honorific
    honorific_avgAge = [
        ("Mr.", 32),
        ("Miss.", 21),
        ("Mrs.", 36),
        ("Master.", 5),
        ("Rare", 45),
    ]
    for t, val in honorific_avgAge:
        t_idx = data["Title"].str.contains(t)
        data["Age"][t_idx].fillna(val, inplace=True)
    honorific_avgFare = [
        ("Mr.", 9),
        ("Miss.", 15),
        ("Mrs.", 26),
        ("Master.", 26),
        ("Rare", 28),
    ]
    for t, val in honorific_avgAge:
        t_idx = data["Title"].str.contains(t)
        data["Fare"][t_idx].fillna(val, inplace=True)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data["Title"] = data["Title"].map(title_mapping)
    # === raw_feature ===
    # Fill null values.
    data["Embarked"].fillna("missing", inplace=True)
    # string feature
    data["Has_Cabin"] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Label encoding.
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2, "missing": 3})
    ticket_uniques = load_pickle("../data/preprocess/ticket_uniques.pkl", verbose=False)
    data["Ticket"] = data["Ticket"].map({u: i for i, u in enumerate(ticket_uniques)})
    cabin_uniques = load_pickle("../data/preprocess/cabin_uniques.pkl", verbose=False)
    data["Cabin"] = data["Cabin"].map({u: i for i, u in enumerate(cabin_uniques)})

    # Drop columns
    data.drop(["Survived", "PassengerId", "Name"], axis=1, inplace=True)
    data = data[
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
            "Has_Cabin",
            "family_size",
            "is_group_guest",
            "ticket_type",
            "Fare_rank",
            "FareRank_Pclass",
            "Age_rank",
            "AgeRank_Pclass",
            "Title",
        ]
    ]
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
