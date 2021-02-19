from mikasa.trainer.nn import TabNetClassificationTrainer, TabNetRegressionTrainer

from .testconf import load_classification_dataset, load_regression_dataset


def test_tabnet_regresser(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    trainer = TabNetRegressionTrainer(
        params={"verbose": 0},
        train_params={"max_epochs": 5, "eval_metric": ["mae"]},
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape


def test_tabnet_classifier(load_classification_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_classification_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    y_train = y_train.ravel()
    y_valid = y_valid.ravel()
    y_test = y_test.ravel()

    trainer = TabNetClassificationTrainer(
        params={"verbose": 0}, train_params={"max_epochs": 5, "eval_metric": ["auc"]}
    )
    trainer.fit(X_train, X_valid, y_train, y_valid)
    y_pred = trainer.predict(X_test)
    assert y_test.shape == y_pred.shape


def test_tabnet_fixed_seed(load_regression_dataset):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = load_regression_dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    def fit_and_prediction(random_state: int = None):
        trainer = TabNetRegressionTrainer(
            params={"verbose": 0},
            train_params={"max_epochs": 5, "eval_metric": ["mae"]},
        )
        trainer.fit(X_train, X_valid, y_train, y_valid, random_state=random_state)
        y_pred = trainer.predict(X_test)
        return y_pred

    # Seed = 0
    y_zero_seed_pred = fit_and_prediction(random_state=0)
    # Seed = 1
    y_one_seed_pred = fit_and_prediction(random_state=1)
    # Seed = 0
    _y_zero_seed_pred = fit_and_prediction(random_state=0)

    assert (y_zero_seed_pred == _y_zero_seed_pred).all()
    assert (y_zero_seed_pred != y_one_seed_pred).all()
