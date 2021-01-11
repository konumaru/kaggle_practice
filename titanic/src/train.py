import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

import config
from mikasa.common import timer
from mikasa.io import load_pickle, dump_pickle
from mikasa.trainer.base import SklearnTrainer
from mikasa.trainer.base import CrossValidationTrainer
from mikasa.trainer.gbdt import LGBMTrainer, XGBTrainer
from mikasa.mlflow_writer import MlflowWriter


def main():
    # Load Data
    feature_files = config.FeatureList.features
    X = load_feature(feature_files)
    y = load_pickle("../data/feature/target.pkl")

    print(X.head())
    print(y.head())

    trainer = SklearnTrainer(config.LogisticParams.model)
    models, oof = run_train(trainer, X, y)
    print(oof)
    # # Set model parameters.
    # lgbm_params = {
    #     "params": config.LightgbmParams.params,
    #     "train_params": config.LightgbmParams.train_params,
    # }
    # xgb_params = {
    #     "params": config.XGBoostPrams.params,
    #     "train_params": config.XGBoostPrams.train_params,
    # }
    # stack_lgbm_params = {
    #     "params": config.StackLightgbmParams.params,
    #     "train_params": config.StackLightgbmParams.train_params,
    # }
    # # Train and Evaluation.
    # with timer("train"):
    #     lgbm_models, lgbm_oof = run_train(LGBMTrainer, lgbm_params, X, y)
    #     xgb_models, xgb_oof = run_train(XGBTrainer, xgb_params, X, y)

    # lgbm_metric = accuracy_score(y, (lgbm_oof > 0.5))
    # xgb_metric = accuracy_score(y, (xgb_oof > 0.5))
    # ensemble_oof = 0.5 * lgbm_oof + 0.5 * xgb_oof
    # ensemble_metric = accuracy_score(y, (ensemble_oof > 0.5))
    # # Stacking
    # X_pred = pd.DataFrame({"lgbm": lgbm_oof, "xgb": xgb_oof})
    # stack_models, stacked_oof = run_train(LGBMTrainer, stack_lgbm_params, X_pred, y)
    # stacked_metric = accuracy_score(y, (stacked_oof > 0.5))
    # # Dump models.
    # dump_pickle(lgbm_models, "../data/working/lgbm_models.pkl")
    # dump_pickle(xgb_models, "../data/working/xgb_models.pkl")
    # dump_pickle(stack_models, "../data/working/stack_models.pkl")
    # # Print metric
    # print(f"Ensemble Metric is {ensemble_metric}")
    # Domp to mlflow.
    # if config.DEBUG is not True:
    #     writer = MlflowWriter(
    #         config.MLflowConfig.experiment_name, tracking_uri="../mlruns"
    #     )
    #     writer.set_run_name(config.MLflowConfig.run_name)
    #     writer.set_note_content(config.MLflowConfig.experiment_note)
    #     writer.log_param("lgbm_params", lgbm_params)
    #     writer.log_param("xgb_params", xgb_params)
    #     writer.log_param("stack_lgbm_params", stack_lgbm_params)
    #     writer.log_param("feature", ", ".join(feature_files))
    #     writer.log_metric("lgbm_auc", lgbm_metric)
    #     writer.log_metric("xgb_auc", xgb_metric)
    #     writer.log_metric("ensemble_auc", ensemble_metric)
    #     writer.log_metric("stacked_auc", stacked_metric)
    #     writer.log_artifact("../data/working/lgbm_models.pkl")
    #     writer.log_artifact("../data/working/xgb_models.pkl")
    #     writer.log_artifact("../data/working/stack_models.pkl")
    #     writer.set_terminated()


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
        params=params,
        train_params=train_params,
        X=X,
        y=y,
    )
    models = cv_trainer.get_models()
    oof = cv_trainer.get_oof()
    return models, oof


if __name__ == "__main__":
    main()
