import os
import sys

sys.path.append("..")

import numpy as np
import pandas as pd

from mikasa.io import save_cache, load_pickle, dump_pickle
from mikasa.preprocessing import add_dummies

dump_dir = "../data/titanic/feature/"


@save_cache(os.path.join(dump_dir, "target.pkl"), use_cache=False)
def target(data: pd.DataFrame):
    target_name = "Survived"
    return data[target_name]


@save_cache(os.path.join(dump_dir, "raw_feature.pkl"), use_cache=False)
def raw_feature(data: pd.DataFrame):
    # Label encoding.
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    data["Embarked"].fillna("missing", inplace=True)
    data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2, "missing": 3})
    # Fill null with average.
    data["Age"].fillna(30, inplace=True)
    data["Fare"].fillna(33, inplace=True)

    # data["Name_length"] = data["Name"].apply(lambda x: len(x.split()))
    # data["Has_Cabin"] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    data["Fare_per_person"] = data.Fare / np.mean(data.SibSp + data.Parch + 1)

    use_cols = [
        "Sex",
        "SibSp",
        "Parch",
        "Age",
        "Fare",
        "Fare_per_person",
        "Pclass",
        "Embarked",
    ]
    return data[use_cols]


@save_cache(os.path.join(dump_dir, "family_feature.pkl"), use_cache=False)
def family_feature(data: pd.DataFrame):
    data["family_size"] = data["SibSp"] + data["Parch"] + 1
    data["is_alone"] = np.where(data["family_size"] == 0, 1, 0)

    dst_cols = ["family_size", "is_alone"]
    return data[dst_cols]


@save_cache(os.path.join(dump_dir, "cabin_feature.pkl"), use_cache=False)
def cabin_feature(data: pd.DataFrame):
    data["Cabin_head"] = data["Cabin"].str.extract(r"(\w)[\d+]", expand=False)
    data["Cabin_head"].fillna("H", inplace=True)

    cabin_head_map = {s: i for i, s in enumerate("ABCDEFGH")}
    data["Cabin_head"] = data["Cabin_head"].map(cabin_head_map)
    return data[["Cabin_head"]]


@save_cache(os.path.join(dump_dir, "fare_rank.pkl"), use_cache=False)
def fare_rank(data: pd.DataFrame):
    data["Fare_rank"] = -1
    data.loc[data["Fare"] <= 7.91, "Fare_rank"] = 0
    data.loc[(data["Fare"] > 7.91) & (data["Fare"] <= 14.454), "Fare_rank"] = 1
    data.loc[(data["Fare"] > 14.454) & (data["Fare"] <= 31), "Fare_rank"] = 2
    data.loc[data["Fare"] > 31, "Fare_rank"] = 3
    data["Fare_rank"] = data["Fare_rank"].astype(int)
    data["FareRank_Pclass"] = data["Fare_rank"] * data["Pclass"]
    return data[["Fare_rank", "FareRank_Pclass"]]


@save_cache(os.path.join(dump_dir, "age_rank.pkl"), use_cache=False)
def age_rank(data: pd.DataFrame):
    data["Age_rank"] = -1
    data.loc[data["Age"] <= 16, "Age_rank"] = 0
    data.loc[(data["Age"] > 16) & (data["Age"] <= 32), "Age_rank"] = 1
    data.loc[(data["Age"] > 32) & (data["Age"] <= 48), "Age_rank"] = 2
    data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age_rank"] = 3
    data.loc[data["Age"] > 64, "Age_rank"] = 4
    data["Age_rank"] = data["Age_rank"].astype(int)
    data["AgeRank_Pclass"] = data["Age_rank"] * data["Pclass"]
    return data[["Age_rank", "AgeRank_Pclass"]]


@save_cache(os.path.join(dump_dir, "name_feature.pkl"), use_cache=False)
def name_feature(data: pd.DataFrame):
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
    data["Age_fillna"] = data["Age"]
    for t, val in honorific_avgAge:
        t_idx = data["Title"].str.contains(t)
        data["Age_fillna"][t_idx].fillna(val, inplace=True)
    data["Age_fillna"].fillna(data["Age"].mean(), inplace=True)
    honorific_avgFare = [
        ("Mr.", 9),
        ("Miss.", 15),
        ("Mrs.", 26),
        ("Master.", 26),
        ("Rare", 28),
    ]
    data["Fare_fillna"] = data["Fare"]
    for t, val in honorific_avgAge:
        t_idx = data["Title"].str.contains(t)
        data["Fare_fillna"][t_idx].fillna(val, inplace=True)
    data["Fare_fillna"].fillna(data["Age"].mean(), inplace=True)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data["Title"] = data["Title"].map(title_mapping)
    return data["Title"]


def create_features(data: pd.DataFrame):
    raw_feature(data.copy())
    family_feature(data.copy())
    cabin_feature(data.copy())
    fare_rank(data.copy())
    age_rank(data.copy())
    name_feature(data.copy())


def main():
    # Load raw data.
    data = pd.read_csv("../data/titanic/raw/train.csv")
    # Save features.
    create_features(data.copy())
    # Dump target
    target(data.copy())


if __name__ == "__main__":
    main()
