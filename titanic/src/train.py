import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

import config
from mikasa.common import timer
from mikasa.io import load_pickle, dump_pickle
from mikasa.trainer.base import SklearnClassificationTrainer
from mikasa.trainer.base import CrossValidationTrainer
from mikasa.trainer.gbdt import LGBMTrainer, XGBTrainer
from mikasa.ensemble import SimpleAgerageEnsember, ManualWeightedEnsember
from mikasa.plot import plot_importance
from mikasa.mlflow_writer import MlflowWriter


def main():
    # Load Data
    feature_files = config.FeatureList.features
    X = load_feature(feature_files)
    y = load_pickle("../data/feature/target.pkl")
    print(X.head())
    print(y.head())

    # >>>>> Fit Trainers.
    # Logistic Regression
    lr_trainer = SklearnClassificationTrainer(config.LogisticRegressionParams.model)
    lr_models, lr_oof, _ = run_train(lr_trainer, X, y)
    lr_accuracy = accuracy_score(y, (lr_oof > 0.5))
    # Rndom Forest
    rf_trainer = SklearnClassificationTrainer(config.RandomForestParams.model)
    rf_models, rf_oof, _ = run_train(rf_trainer, X, y)
    rf_accuracy = accuracy_score(y, (rf_oof > 0.5))
    # LightGBM
    lgbm_trainer = LGBMTrainer(
        config.LightgbmParams.params, config.LightgbmParams.train_params
    )
    lgbm_models, lgbm_oof, lgbm_importance_fig = run_train(lgbm_trainer, X, y)
    lgbm_accuracy = accuracy_score(y, (lgbm_oof > 0.5))
    # XGBoost
    xgb_trainer = XGBTrainer(
        config.XGBoostPrams.params, config.XGBoostPrams.train_params
    )
    xgb_models, xgb_oof, xgb_importance_fig = run_train(xgb_trainer, X, y)
    xgb_accuracy = accuracy_score(y, (xgb_oof > 0.5))

    # >>>>> Print Metric.
    print(f"{'LR':>6} Accuracy is {lr_accuracy:.08f}")
    print(f"{'RF':>6} Accuracy is {rf_accuracy:.08f}")
    print(f"{'LGBM':>6} Accuracy is {lgbm_accuracy:.08f}")
    print(f"{'XGB':>6} Accuracy is {xgb_accuracy:.08f}")

    # >>>>> Ensemble.
    oof_dict = {
        "LR": lr_oof,
        "RF": rf_oof,
        "LGBM": lgbm_oof,
        "XGB": xgb_oof,
    }
    oof_df = pd.DataFrame(oof_dict)
    print(oof_df.head())
    ensembler = ManualWeightedEnsember(weights=[0.1, 0.1, 0.2, 0.6])
    ensembler.fit(oof_df.to_numpy(), y)
    ensemble_oof = ensembler.predict(oof_df.to_numpy())
    ensemble_accuracy = accuracy_score(y, (ensemble_oof > 0.5))
    print(
        "{0:>6} Accuracy is {1:.08f}".format(
            "SimpleAgerageEnsember",
            ensemble_accuracy,
        )
    )

    # >>>>> Dump models.
    dump_pickle(lgbm_models, "../data/working/lr_models.pkl")
    dump_pickle(lgbm_models, "../data/working/rf_models.pkl")
    dump_pickle(lgbm_models, "../data/working/lgbm_models.pkl")
    dump_pickle(xgb_models, "../data/working/xgb_models.pkl")

    # >>>> Domp to mlflow.
    if config.DEBUG is not True:
        writer = MlflowWriter(
            config.MLflowConfig.experiment_name, tracking_uri="../mlruns"
        )
        writer.set_run_name(config.MLflowConfig.run_name)
        writer.set_note_content(config.MLflowConfig.experiment_note)
        # Logistic Regression
        writer.log_param("LR_params", config.LogisticRegressionParams.params)
        writer.log_metric("LR_Accuracy", lr_accuracy)
        writer.log_artifact("../data/working/lr_models.pkl")
        # Rndom Forest
        writer.log_param("RF_params", config.RandomForestParams.params)
        writer.log_metric("RF_Accuracy", rf_accuracy)
        writer.log_artifact("../data/working/rf_models.pkl")
        # LightGBM
        writer.log_param(
            "LGBM_params",
            {
                "params": config.LightgbmParams.params,
                "train_params": config.LightgbmParams.train_params,
            },
        )
        writer.log_metric("LGBM_Accuracy", lgbm_accuracy)
        writer.log_artifact("../data/working/lgbm_models.pkl")
        writer.log_figure(lgbm_importance_fig, "lgbm_importance.png")
        lgbm_importance_fig.savefig("../data/working/lgbm_importance.png")
        # XGBoost
        writer.log_param(
            "XGB_params",
            {
                "params": config.XGBoostPrams.params,
                "train_params": config.XGBoostPrams.train_params,
            },
        )
        writer.log_metric("XGB_Accuracy", xgb_accuracy)
        writer.log_artifact("../data/working/xgb_models.pkl")
        writer.log_figure(xgb_importance_fig, "xgb_importance.png")
        xgb_importance_fig.savefig("../data/working/xgb_importance.png")
        # Ensemble
        writer.log_metric("Ensemble_Accuracy", ensemble_accuracy)
        # Close writer client.
        writer.set_terminated()


def load_feature(feature_files):
    feature_files = [f"../data/feature/{filename}.pkl" for filename in feature_files]
    data = []
    for filepath in feature_files:
        feature = load_pickle(filepath)
        data.append(feature)
    feature = pd.concat(data, axis=1)
    return feature


def run_train(Trainer, X, y, params=None, train_params=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    cv_trainer = CrossValidationTrainer(cv, Trainer)
    cv_trainer.fit(
        X=X,
        y=y,
    )
    models = cv_trainer.get_models()
    oof = cv_trainer.get_oof()

    if "Sklearn" not in Trainer.__class__.__name__:
        (name, mean_importance, std_importance) = cv_trainer.get_importance(
            max_feature=50
        )
        importance_fig = plot_importance(name, mean_importance, std_importance)
    else:
        importance_fig = plt.figure()

    return models, oof, importance_fig


if __name__ == "__main__":
    with timer("Train"):
        main()
