import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from mikasa.io import save_cache, load_pickle, dump_pickle

dump_dir = "../data/feature/"


@save_cache(os.path.join(dump_dir, "target.pkl"), use_cache=False)
def target():
    data = pd.read_csv("../data/raw/train.csv")
    target_name = "Survived"
    return data[target_name]


@save_cache(os.path.join(dump_dir, "family_group.pkl"), use_cache=False)
def family_feature():
    data = pd.read_csv("../data/raw/train.csv")
    data["family_size"] = data["SibSp"] + data["Parch"] + 1
    data["is_group_guest"] = np.where(data["family_size"] >= 1, 1, 0)

    # data["escape_boarding_proba"] = np.where(data["is_group_guest"] == 0, 1, -1)
    # for t in ["Mr.", "Miss.", "Mrs.", "Master."]:
    #     t_idx = data["Name"].str.contains(t)
    #     data["escape_boarding_proba"][t_idx] = 1 / data["family_size"][t_idx]

    dst_cols = [
        "family_size",
        "is_group_guest",
        # "escape_boarding_proba",
    ]
    print(data[dst_cols].head())
    return data[dst_cols]


@save_cache(os.path.join(dump_dir, "raw_feature.pkl"), use_cache=False)
def raw_feature():
    data = pd.read_csv("../data/raw/train.csv")
    # Fill null values.
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

    # Drop columns
    data.drop(["Survived", "PassengerId", "Name"], axis=1, inplace=True)
    print(data.head())
    return data


def create_features():
    target()
    # Features
    raw_feature()
    family_feature()


def main():
    train = pd.read_csv("../data/raw/train.csv")
    # Dump unique values.
    ticket_uniques = train["Ticket"].fillna("missing").to_numpy()
    dump_pickle(ticket_uniques, "../data/preprocess/ticket_uniques.pkl", verbose=False)

    cabin_uniques = train["Cabin"].fillna("missing").to_numpy()
    dump_pickle(cabin_uniques, "../data/preprocess/cabin_uniques.pkl", verbose=False)

    create_features()


if __name__ == "__main__":
    main()
