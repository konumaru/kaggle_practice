import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../..")

from utils import read_train, fast_merge


def add_virtual_timestamp(train):
    """Add converted from timestamp to vritual timestamp.
    notebook: https://www.kaggle.com/marisakamozz/cv-strategy-in-the-kaggle-environment

    Parameters
    ----------
    train : pd.DataFrame
        raw data of train.csv

    Returns
    -------
    train : pd.DataFrame
    """

    a = 2.2
    b = 2.3

    max_timestamp_u = train[["user_id", "timestamp"]].groupby(["user_id"]).max()
    max_timestamp_u.columns = ["max_timestamp"]
    max_timestamp_u["interval"] = (
        max_timestamp_u.max_timestamp.max() - max_timestamp_u.max_timestamp
    )

    max_timestamp_u["random"] = np.random.beta(a, b, len(max_timestamp_u))
    max_timestamp_u["random_timestamp"] = (
        max_timestamp_u.interval * max_timestamp_u.random
    )
    max_timestamp_u["random_timestamp"] = max_timestamp_u.random_timestamp.astype(int)
    max_timestamp_u.drop(["interval", "random"], axis=1, inplace=True)

    train = fast_merge(train, max_timestamp_u, "user_id")
    train["virtual_timestamp"] = train.timestamp + train.random_timestamp
    train.set_index(["virtual_timestamp", "row_id"], inplace=True)
    train.sort_index(inplace=True)
    train.reset_index(inplace=True)
    train.drop(columns=["max_timestamp", "random_timestamp"], inplace=True)
    return train


def main():
    train = read_train()
    train = add_virtual_timestamp(train)

    val_size = 2500000
    for cv in range(5):
        valid = train[-val_size:]
        train = train[:-val_size]
        valid.to_pickle(f"../data/working/cv{cv+1}_valid.pickle")
        train.to_pickle(f"../data/working/cv{cv+1}_train.pickle")

        print(f"[{cv+1}]")
        train_user = train.user_id.unique()
        valid_user = valid.user_id.unique()
        new_user = np.setdiff1d(valid_user, train_user)
        print(
            f"new user in valid:{len(new_user)}, train:{len(train_user)}, valid:{len(valid_user)}"
        )

        train_content = train.content_id.unique()
        valid_content = valid.content_id.unique()
        new_content = np.setdiff1d(valid_content, train_content)
        print(
            f"new content in valid:{len(new_content)}, train:{len(train_content)}, valid:{len(valid_content)}"
        )


if __name__ == "__main__":
    main()
