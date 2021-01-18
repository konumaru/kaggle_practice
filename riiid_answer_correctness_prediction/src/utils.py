import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd


def read_train():
    cache_path = "../data/working/train.pkl"
    train_path = "../data/raw/train.csv"

    dtypes_train = {
        "row_id": "int64",
        "timestamp": "int64",
        "user_id": "int32",
        "content_id": "int16",
        "content_type_id": "int8",
        "task_container_id": "int16",
        "user_answer": "int8",
        "answered_correctly": "int8",
        "prior_question_elapsed_time": "float32",
        "prior_question_had_explanation": "boolean",
    }

    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    else:
        train = pd.read_csv(train_path, dtype=dtypes_train)
        train.to_pickle(cache_path)
        return train


# "FAST PANDAS LEFT JOIN (357x faster than pd.merge)" by tkm2261
# https://www.kaggle.com/tkm2261/fast-pandas-left-join-357x-faster-than-pd-merge
def fast_merge(left, right, key):
    return pd.concat(
        [
            left.reset_index(drop=True),
            right.reindex(left[key].values).reset_index(drop=True),
        ],
        axis=1,
    )
