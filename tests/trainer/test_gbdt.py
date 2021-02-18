from mikasa.trainer.gbdt import XGBTrainer, LGBMTrainer, CatBoostTrainer

from .testconf import load_classification_dataset, load_regression_dataset


def test_xgb_regresser(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    trainer = XGBTrainer(
        params={"objective": "reg:squarederror", "eval_metric": "rmse"},
        train_params={"verbose_eval": False},
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape

    importance = trainer.get_importance()
    assert type(importance) == dict


def test_xgb_classifier(load_classification_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_classification_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    trainer = XGBTrainer(
        params={"objective": "binary:logistic", "eval_metric": "auc"},
        train_params={"verbose_eval": False},
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape

    importance = trainer.get_importance()
    assert type(importance) == dict


def test_xgb_fixed_seed(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    def fit_and_prediction(random_state: int = 0):
        trainer = XGBTrainer(
            params={
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "colsample_bytree": 0.7,
            },
            train_params={"verbose_eval": False},
        )
        trainer.fit(X_train, X_valid, y_train, y_valid, random_state=random_state)
        pred = trainer.predict(X_test)
        return pred

    # Seed = 0
    y_zero_seed_pred = fit_and_prediction(random_state=0)
    # Seed = 1
    y_one_seed_pred = fit_and_prediction(random_state=1)
    # Seed = 0
    _y_zero_seed_pred = fit_and_prediction(random_state=0)

    assert (y_zero_seed_pred == _y_zero_seed_pred).all()
    assert (y_zero_seed_pred != y_one_seed_pred).all()


def test_lgbm_regresser(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    trainer = LGBMTrainer(
        params={"objective": "regression", "metric": "rmse", "verbosity": 0},
        train_params={},
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape

    importance = trainer.get_importance()
    assert type(importance) == dict


def test_lgbm_classifier(load_classification_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_classification_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    trainer = LGBMTrainer(
        params={"objective": "binary", "metric": "auc", "verbosity": 0},
        train_params={},
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape

    importance = trainer.get_importance()
    assert type(importance) == dict


def test_lgbm_fixed_seed(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2
    )

    def fit_and_prediction(random_state: int = None):
        trainer = LGBMTrainer(
            params={
                "objective": "regression",
                "metric": "rmse",
                "feature_fraction": 0.7,
            },
            train_params={},
        )
        trainer.fit(X_train, X_valid, y_train, y_valid, random_state=random_state)
        pred = trainer.predict(X_test)
        return pred

    # Seed = 0
    y_zero_seed_pred = fit_and_prediction(random_state=0)
    # Seed = 1
    y_one_seed_pred = fit_and_prediction(random_state=1)
    # Seed = 0
    _y_zero_seed_pred = fit_and_prediction(random_state=0)

    assert (y_zero_seed_pred == _y_zero_seed_pred).all()
    assert (y_zero_seed_pred != y_one_seed_pred).all()


def test_cat_regresser(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    trainer = CatBoostTrainer(
        params={"loss_function": "MAE", "verbose": 0},
        train_params={},
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape

    importance = trainer.get_importance()
    assert type(importance) == dict


def test_cat_classifier(load_classification_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_classification_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    trainer = CatBoostTrainer(
        params={"loss_function": "Logloss", "verbose": 0},
        train_params={},
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape

    importance = trainer.get_importance()
    assert type(importance) == dict


def test_cat_fixed_seed(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2
    )

    def fit_and_prediction(random_state: int = None):
        trainer = CatBoostTrainer(
            params={
                "iterations": 100,
                "loss_function": "MAE",
                "sampling_frequency": "PerTreeLevel",
                "subsample": 0.7,
                "colsample_bylevel": 0.7,
                "save_snapshot": False,
                "verbose": 0,
            },
            train_params={},
        )
        trainer.fit(X_train, X_valid, y_train, y_valid, random_state=random_state)
        pred = trainer.predict(X_test)
        return pred

    # Seed = 0
    y_zero_seed_pred = fit_and_prediction(random_state=0)
    # Seed = 1
    y_one_seed_pred = fit_and_prediction(random_state=1)
    # Seed = 0
    _y_zero_seed_pred = fit_and_prediction(random_state=0)

    assert (y_zero_seed_pred == _y_zero_seed_pred).all()
    assert (y_zero_seed_pred != y_one_seed_pred).all()
