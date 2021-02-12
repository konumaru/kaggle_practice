import numpy as np
from sklearn.metrics import accuracy_score

DEBUG = False

SEED = 42
NUM_SEED = 3


class MLflowConfig:
    experiment_name = "Titanic, LGBM"
    run_name = "From Add cabin_feature"
    experiment_note = """
    Convert one-hot to ordinal encode
    """


class FeatureList:
    features = [
        "raw_feature",
        "cabin_feature",
        # >>>>> Not change accuracy features <<<<<
        # >>>>> Not improved features <<<<<
        # "age_rank",
        # "family_feature",
        # "fare_rank",
        # "name_feature",
    ]


def accuracy(preds, data):
    """精度 (Accuracy) を計算する関数"""
    # 正解ラベル
    y_true = data.get_label()
    preds = np.where(preds > 0.5, 1, 0)
    acc = accuracy_score(y_true, preds)
    return "accuracy", acc, True


class LightgbmParams:

    params = {
        "objective": "binary",
        "metric": "None",
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
        "feval": accuracy,
    }
