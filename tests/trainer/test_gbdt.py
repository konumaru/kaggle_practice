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
