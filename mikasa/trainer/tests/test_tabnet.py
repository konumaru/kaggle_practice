import pytest

import pandas as pd

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mikasa.trainer.nn import TabNetClassificationTrainer


def test_tabnet_classification():
    X, Y = make_classification(
        random_state=12,
        n_samples=10_000,
        n_features=100,
        n_redundant=3,
        n_informative=20,
        n_clusters_per_class=1,
        n_classes=2,
    )

    X, X_test, Y, y_test = train_test_split(X, Y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y)

    X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
    X_valid, y_valid = pd.DataFrame(X_valid), pd.DataFrame(y_valid)
    X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)

    trainer = TabNetClassificationTrainer()
    trainer.fit({}, {}, X_train, y_train, X_valid, y_valid, y_train)
