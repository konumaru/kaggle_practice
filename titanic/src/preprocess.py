import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from mikasa.io import save_cache, load_pickle, dump_pickle
from mikasa.preprocessing import add_dummies

dump_dir = "../data/feature/"


@save_cache(os.path.join(dump_dir, "target.pkl"), use_cache=False)
def target():
    data = pd.read_csv("../data/raw/train.csv")
    target_name = "Survived"
    return data[target_name]


@save_cache(os.path.join(dump_dir, "raw_feature.pkl"), use_cache=False)
def raw_feature():
    data = pd.read_csv("../data/raw/train.csv")
    # Label encoding.
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    # One-hot encoding
    data = add_dummies(data, "Pclass")
    data["Embarked"].fillna("missing", inplace=True)
    data = add_dummies(data, "Embarked")
    # Fill null with average.
    data["Age"].fillna(30, inplace=True)
    data["Fare"].fillna(33, inplace=True)

    use_cols = ["Sex", "SibSp", "Parch", "Age", "Fare"]
    use_cols += list(data.columns[data.columns.str.contains("Pclass")])
    use_cols += list(data.columns[data.columns.str.contains("Embarked")])
    return data[use_cols]


@save_cache(os.path.join(dump_dir, "family_feature.pkl"), use_cache=False)
def family_feature():
    data = pd.read_csv("../data/raw/train.csv")
    data["family_size"] = data["SibSp"] + data["Parch"] + 1
    data["is_alone"] = np.where(data["family_size"] == 0, 1, 0)

    dst_cols = [
        "family_size",
        "is_alone",
    ]
    return data[dst_cols]


@save_cache(os.path.join(dump_dir, "ticket_type.pkl"), use_cache=False)
def ticket_type():
    data = pd.read_csv("../data/raw/train.csv")
    data["ticket_type"] = data["Ticket"].apply(lambda x: x[0:3])
    cabin_uniques = load_pickle(
        "../data/preprocess/ticketType_uniques.pkl", verbose=False
    )
    data["ticket_type"] = data["ticket_type"].map(
        {u: i for i, u in enumerate(cabin_uniques)}
    )
    print(data[["ticket_type"]].head())
    return data[["ticket_type"]]


@save_cache(os.path.join(dump_dir, "fare_rank.pkl"), use_cache=False)
def fare_rank():
    data = pd.read_csv("../data/raw/train.csv")
    data["Fare_rank"] = -1
    data.loc[data["Fare"] <= 7.91, "Fare_rank"] = 0
    data.loc[(data["Fare"] > 7.91) & (data["Fare"] <= 14.454), "Fare_rank"] = 1
    data.loc[(data["Fare"] > 14.454) & (data["Fare"] <= 31), "Fare_rank"] = 2
    data.loc[data["Fare"] > 31, "Fare_rank"] = 3
    data["Fare_rank"] = data["Fare_rank"].astype(int)
    data["FareRank_Pclass"] = data["Fare_rank"] * data["Pclass"]
    return data[["Fare_rank", "FareRank_Pclass"]]


@save_cache(os.path.join(dump_dir, "name_feature.pkl"), use_cache=False)
def name_feature():
    data = pd.read_csv("../data/raw/train.csv")
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
    return data[["Title"]]


@save_cache(os.path.join(dump_dir, "age_rank.pkl"), use_cache=False)
def age_rank():
    data = pd.read_csv("../data/raw/train.csv")
    data["Age_rank"] = -1
    data.loc[data["Age"] <= 16, "Age_rank"] = 0
    data.loc[(data["Age"] > 16) & (data["Age"] <= 32), "Age_rank"] = 1
    data.loc[(data["Age"] > 32) & (data["Age"] <= 48), "Age_rank"] = 2
    data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age_rank"] = 3
    data.loc[data["Age"] > 64, "Age_rank"] = 4
    data["Age_rank"] = data["Age_rank"].astype(int)
    data["AgeRank_Pclass"] = data["Age_rank"] * data["Pclass"]
    return data[["Age_rank", "AgeRank_Pclass"]]


def create_features():
    target()
    # Features
    raw_feature()
    family_feature()
    # ticket_type()
    # fare_rank()
    # age_rank()
    # name_feature()


def main():
    create_features()


if __name__ == "__main__":
    main()
