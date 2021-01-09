class FeatureList:
    features = [
        "raw_feature.pkl",
    ]


class LightgbmParams:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 300,
        "learning_rate": 0.1,
        "max_depth": 2,
        "random_seed": 42,
        "verbose": -1,
    }
    train_params = {
        "verbose_eval": 10,
        "num_boost_round": 1000,
        "early_stopping_rounds": 10,
    }


class XGBoostPrams:
    params = {
        "objective": "binary:logistic",
        "metric": "logloss",
        "learning_rate": 0.1,
        "random_seed": 42,
        "max_depth": 5,
        "gammma": 0.1,
        "colsample_bytree": 1,
        "min_child_weight": 1,
        "seed": 42,
        "verbose": -1,
    }
    train_params = {
        "verbose_eval": 10,
        "num_boost_round": 500,
        "early_stopping_rounds": 10,
    }


class StackLightgbmParams:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 100,
        "learning_rate": 0.1,
        "max_depth": 2,
        "random_seed": 42,
        "verbose": -1,
    }
    train_params = {
        "verbose_eval": 10,
        "num_boost_round": 500,
        "early_stopping_rounds": 10,
    }
