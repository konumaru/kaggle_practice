import numpy as np
import pandas as pd

from mikasa.trainer.base import SklearnRegressionTrainer
from mikasa.trainer.cross_validation import CrossValidationTrainer, RSACVTrainer

from .testconf import load_regression_dataset


def test_cv_trainer(load_regression_dataset):
    from sklearn import metrics
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = load_regression_dataset

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train.ravel())
    y_test = pd.Series(y_test.ravel())

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    base_trainer = SklearnRegressionTrainer(
        model=RandomForestRegressor(n_estimators=10)
    )
    trainer = CrossValidationTrainer(cv, base_trainer)
    trainer.fit(X_train, y_train)
    y_pred = np.mean(trainer.predict(X_test), axis=0)
    assert y_test.shape == y_pred.shape

    score = metrics.mean_squared_error(y_test, y_pred)
    # Get importance
    (name, mean_importance, std_importance) = trainer.get_importance()
    # Get models.
    models = trainer.get_model()
    # Get oof and target.
    oof = trainer.get_cv_oof()
    target = trainer.get_cv_targets()
    assert oof.shape == target.shape
    score = metrics.mean_squared_error(oof, target)


def test_rsa_cv_trainer(load_regression_dataset):
    from sklearn import metrics
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = load_regression_dataset

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train.ravel())
    y_test = pd.Series(y_test.ravel())

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    base_trainer = SklearnRegressionTrainer(
        model=RandomForestRegressor(n_estimators=10)
    )
    trainer = RSACVTrainer(cv, base_trainer)
    trainer.fit(X_train, y_train, random_state=42, num_seed=3)
    y_pred = np.mean(trainer.predict(X_test), axis=0)
    assert y_test.shape == y_pred.shape

    score = metrics.mean_squared_error(y_test, y_pred)
    # Get importance
    (name, mean_importance, std_importance) = trainer.get_importance()
    # Get models.
    models = trainer.get_model()
    # Get oof and target.
    oof = trainer.get_cv_oof()
    target = trainer.get_cv_targets()
    assert oof.shape == target.shape
    score = metrics.mean_squared_error(oof, target)
