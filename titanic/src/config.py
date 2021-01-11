DEBUG = False


class MLflowConfig:
    experiment_name = "LightGBM+XGboost_StackedLightGBM"
    run_name = "add name_feature"
    experiment_note = """
    """


class FeatureList:
    features = [
        "raw_feature",
        "family_group",
        "ticket_type",
        "fare_rank",
        "age_rank",
        "name_feature",
    ]


class LightgbmParams:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
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
