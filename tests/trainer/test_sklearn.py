import numpy as np
from sklearn import metrics

from mikasa.trainer.base import SklearnRegressionTrainer
from mikasa.trainer.base import SklearnClassificationTrainer
from .testconf import load_classification_dataset, load_regression_dataset


def test_sklearn_regression_trainer(load_regression_dataset):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge, LinearRegression

    X_train, X_test, y_train, y_test = load_regression_dataset

    models = [
        RandomForestRegressor(n_estimators=10),
        LinearRegression(),
        Ridge(),
    ]
    for model in models:
        trainer = SklearnRegressionTrainer(model=model)
        trainer.fit(X_train, y_train)

        y_pred = trainer.predict(X_test)
        score = metrics.mean_squared_error(y_test, y_pred)

        assert y_test.shape == y_pred.shape


def test_sklearn_classification_trainer(load_classification_dataset):
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = load_classification_dataset

    models = [
        SVC(probability=True),
        LogisticRegression(),
        RandomForestClassifier(n_estimators=10),
    ]
    for model in models:
        trainer = SklearnClassificationTrainer(model=model)
        trainer.fit(X_train, y_train)

        y_pred = trainer.predict(X_test)
        score = metrics.roc_auc_score(y_test, y_pred)

        assert y_test.shape == y_pred.shape


def test_seed_fixed(load_regression_dataset):
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = load_regression_dataset
    model = RandomForestRegressor(n_estimators=10)

    # Seed = 0
    trainer = SklearnRegressionTrainer(model=model)
    trainer.fit(X_train, y_train, random_state=0)
    y_zero_seed_pred = trainer.predict(X_test)

    # Seed = 1
    trainer = SklearnRegressionTrainer(model=model)
    trainer.fit(X_train, y_train, random_state=1)
    y_one_seed_pred = trainer.predict(X_test)

    # Seed = 0
    trainer = SklearnRegressionTrainer(model=model)
    trainer.fit(X_train, y_train, random_state=0)
    _y_zero_seed_pred = trainer.predict(X_test)

    assert (y_zero_seed_pred == _y_zero_seed_pred).all()
    assert (y_zero_seed_pred != y_one_seed_pred).all()
