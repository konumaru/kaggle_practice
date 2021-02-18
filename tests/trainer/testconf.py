import pytest


@pytest.fixture
def load_classification_dataset():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    seed = 12
    X, Y = make_classification(
        random_state=seed,
        n_samples=10_000,
        n_features=100,
        n_redundant=3,
        n_informative=20,
        n_clusters_per_class=1,
        n_classes=2,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def load_regression_dataset():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    seed = 12
    X, Y = make_regression(
        random_state=seed,
        n_samples=10_000,
        n_features=100,
        n_informative=10,
        n_targets=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, y_train, y_test
