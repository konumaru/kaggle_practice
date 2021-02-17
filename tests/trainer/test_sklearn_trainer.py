import pytest

import numpy as np

from testconf import load_classification_dataset


def test_foo(load_classification_dataset):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_classification_dataset
    assert X_train.shape[1] == X_valid.shape[1]
