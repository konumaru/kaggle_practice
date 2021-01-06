import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from mikasa.io import save_cache, load_pickle, dump_pickle

dump_dir = "../data/feature/"


@save_cache(os.path.join(dump_dir, "target.pkl"), use_cache=False)
def extract_target(data):
    target_name = "Survived"


@save_cache(os.path.join(dump_dir, "raw_feature.pkl"), use_cache=False)
def extract_raw_feature(data):
    # Fill null values.
    data["Embarked"].fillna("missing", inplace=True)

    # Label encoding.
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2, "missing": 3})
    ticket_uniques = load_pickle("../data/preprocess/ticket_uniques.pkl", verbose=False)
    data["Ticket"] = data["Ticket"].map({u: i for i, u in enumerate(ticket_uniques)})
    cabin_uniques = load_pickle("../data/preprocess/cabin_uniques.pkl", verbose=False)
    data["Cabin"] = data["Cabin"].map({u: i for i, u in enumerate(cabin_uniques)})

    return data


def create_features(data: pd.DataFrame):
    extract_target(data)
    extract_raw_feature(data)


def main():
    train = pd.read_csv("../data/raw/train.csv")
    # Dump unique values.
    ticket_uniques = train["Ticket"].fillna("missing").to_numpy()
    dump_pickle(ticket_uniques, "../data/preprocess/ticket_uniques.pkl", verbose=False)

    cabin_uniques = train["Cabin"].fillna("missing").to_numpy()
    dump_pickle(cabin_uniques, "../data/preprocess/cabin_uniques.pkl", verbose=False)

    create_features(train)


if __name__ == "__main__":
    main()
