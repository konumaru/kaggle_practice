from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DEBUG = True


class MLflowConfig:
    experiment_name = "EnsembleModel"
    run_name = "Metric Change for AUC"
    experiment_note = """
    """


class FeatureList:
    features = [
        "raw_feature",
        # "family_feature",
        # "fare_rank",
        # "age_rank",
        # "name_feature",
        # "multi_feature",
        # "cabin_feature",
    ]


class LogisticRegressionParams:
    params = {
        "penalty": "l2",
        "C": 1.0,
        "class_weight": None,
        "random_state": 42,
        "solver": "lbfgs",
        "max_iter": 500,
        "warm_start": False,
        "n_jobs": -1,
    }
    model = LogisticRegression(**params)


class RandomForestParams:
    params = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": 5,
        "max_samples": 0.8,
        "min_samples_split": 10,
        "min_samples_leaf": 10,
        "random_state": 42,
        "warm_start": False,
        "class_weight": None,  # balanced, balanced_subsample
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)


class LightgbmParams:
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 300,
        "learning_rate": 0.1,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "scale_pos_weight": 1,
        "max_depth": 2,
        "random_seed": 42,
        "verbose": -1,
    }
    train_params = {
        "verbose_eval": 10,
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
    }


class XGBoostPrams:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.1,
        "max_depth": 4,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "scale_pos_weight": 1,
        "gamma": 0.9,
        "subsample": 0.8,
        "nthread": -1,
        "seed": 42,
    }
    train_params = {
        "verbose_eval": 10,
        "num_boost_round": 2000,
        "early_stopping_rounds": 50,
    }
