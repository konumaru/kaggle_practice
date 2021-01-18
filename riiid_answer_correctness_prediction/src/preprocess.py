import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from mikasa.common import timer

from utils import read_train


def main():
    # TODO:
    # raw data を読み込む
    # cv split をする
    # Dataset を定義
    # Dataloader に変換
    # next(iter(dataloader)) でデータを取得できることを確認

    train = read_train()
    print(train.head())


if __name__ == "__main__":
    with timer("Preprocessing"):
        main()
