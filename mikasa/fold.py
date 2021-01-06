import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold


class StratifiedGroupKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y: np.ndarray = None, groups=None):
        fold = pd.DataFrame([y, groups]).T
        fold.columns = ["y", "groups"]
        fold["y"] = fold["y"].astype(int)
        g = fold.groupby("groups")["y"].agg("mean").reset_index()
        fold = fold.merge(g, how="left", on="groups", suffixes=("", "_mean"))
        fold["y_mean"] = fold["y_mean"].apply(np.round)
        fold["fold_id"] = 0
        for unique_y in fold["y_mean"].unique():
            mask = fold.y_mean == unique_y
            selected = fold[mask].reset_index(drop=True)
            cv = GroupKFold(n_splits=self.n_splits)
            for i, (train_index, valid_index) in enumerate(
                cv.split(range(len(selected)), y=None, groups=selected["groups"])
            ):
                selected.loc[valid_index, "fold_id"] = i
            fold.loc[mask, "fold_id"] = selected["fold_id"].values

        for i in range(self.n_splits):
            indices = np.arange(len(fold))
            train_index = indices[fold["fold_id"] != i]
            valid_index = indices[fold["fold_id"] == i]
            yield train_index, valid_index
