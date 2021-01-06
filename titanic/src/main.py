import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from mikasa.common import timer

# TODO:
# Name以外のカラムを前処理
# StratifiedKFoldをつかってLightGBMで学習
# oof のスコアでモデルを評価

# =========================================


def preprocess(data):
    drop_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
    data.drop(drop_cols, axis=1, inplace=True)
    return data


# =========================================


def train(data):
    model = None
    metric = None
    return model, metric


# =========================================


def submission(model):
    return None


def main():
    with timer("Load and Preprocess"):
        train = pd.read_csv("../data/raw/train.csv")
        train = preprocess(train)

    print(train.head())


if __name__ == "__main__":
    main()
