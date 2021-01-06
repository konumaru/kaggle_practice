import os
import numpy as np
import pandas as pd

from mikasa.common import timer


def submission(models, data):
    def lgbm_predict(models, data):
        preds = [m.predict(data) for m in models]
        return np.mean(preds, axis=0)

    pred = lgbm_predict(models, data)
    pred = (pred > 0.5).astype(np.int8)
    return pred


def main():
    with timer("Submission"):
        pred = submission(models, test[features])

        test = pd.read_csv("../data/raw/test.csv")
        submit = test[["PassengerId"]].copy()
        submit["Survived"] = pred
        submit.to_csv("../data/submit/submission.csv", index=False)
        print(submit.head())


if __name__ == "__main__":
    main()
