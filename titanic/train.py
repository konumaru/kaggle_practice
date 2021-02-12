import os
import sys

sys.path.append("..")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

import config
from mikasa.common import timer
from mikasa.io import load_pickle, dump_pickle, load_feature

from mikasa.trainer.gbdt import LGBMTrainer
from mikasa.trainer.cross_validation import RSACrossValidationTrainer
from mikasa.trainer.base import SklearnRegressionTrainer

from mikasa.plot import plot_importance
from mikasa.mlflow_writer import MlflowWriter


def run_train(model_name, cv, base_trainer, X, y):
    trainer = RSACrossValidationTrainer(cv, base_trainer, seed=config.SEED)
    trainer.fit(X=X, y=y, num_seed=config.NUM_SEED)
    # Save model.
    models = trainer.get_models()
    dump_pickle(models, f"../data/titanic/model/{model_name}_models.pkl")
    # Evaluation by cv.
    oof = trainer.get_oof()
    is_usage_oof = np.logical_not(np.isnan(oof))
    oof = np.where(oof > 0.5, 1, 0)
    metric = accuracy_score(y[is_usage_oof], oof[is_usage_oof])
    return trainer, metric


def main():
    # Load data.
    src_dir = "../data/titanic/feature/"
    feature_files = config.FeatureList.features
    feature_files = [
        os.path.join(src_dir, f"{filename}.pkl") for filename in feature_files
    ]
    X = load_feature(feature_files)
    y = load_pickle(os.path.join(src_dir, "target.pkl"))
    print(X.head())
    print(y.head())

    # Split data
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )

    metric_dict = {}

    # Train model.
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    lgbm_trainer = LGBMTrainer(
        config.LightgbmParams.params, config.LightgbmParams.train_params
    )
    trainer, metric = run_train("LGBM", cv, lgbm_trainer, X_train, y_train)
    metric_dict["LGBM"] = metric
    # Save figure of feature importance.
    name, mean_importance, std_importance = trainer.get_importance()
    fig = plot_importance(name, mean_importance, std_importance)
    fig.savefig("../data/titanic/working/lgbm_importance.png")

    # Prediction
    pred_first = []
    predict = np.array(trainer.predict(X_eval)).T
    pred_first.append(predict)
    pred_first = np.concatenate(pred_first, axis=1)
    pred_first = pd.DataFrame(pred_first)

    # Stacking
    base_trainer = SklearnRegressionTrainer(model=Ridge(random_state=config.SEED))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    trainer, metric = run_train("stack_ridge", cv, base_trainer, pred_first, y_eval)
    print(trainer.get_models()[0].coef_)
    metric_dict["Stack"] = metric

    # Evaluation
    for model_name, metric in metric_dict.items():
        print(f"{model_name:>8} Metric: {metric:.08f}")

    # Domp logs to mlflow.
    if config.DEBUG is not True:
        writer = MlflowWriter(
            config.MLflowConfig.experiment_name,
            tracking_uri=os.path.abspath("../mlruns"),
        )
        writer.set_run_name(config.MLflowConfig.run_name)
        writer.set_note_content(config.MLflowConfig.experiment_note)
        # Features
        writer.log_param("Feature", ", ".join(feature_files))
        # Logistic Regression
        # writer.log_param("LR_params", config.LogisticRegressionParams.params)
        # writer.log_metric("LR_Accuracy", lr_accuracy)
        # writer.log_artifact("../data/working/lr_models.pkl")
        # # Rndom Forest
        # writer.log_param("RF_params", config.RandomForestParams.params)
        # writer.log_metric("RF_Accuracy", rf_accuracy)
        # writer.log_artifact("../data/working/rf_models.pkl")
        # # LightGBM
        # writer.log_param(
        #     "LGBM_params",
        #     {
        #         "params": config.LightgbmParams.params,
        #         "train_params": config.LightgbmParams.train_params,
        #     },
        # )
        # writer.log_metric("LGBM_Accuracy", lgbm_accuracy)
        # writer.log_artifact("../data/working/lgbm_models.pkl")
        # writer.log_figure(lgbm_importance_fig, "lgbm_importance.png")
        # lgbm_importance_fig.savefig("../data/working/lgbm_importance.png")
        # # XGBoost
        # writer.log_param(
        #     "XGB_params",
        #     {
        #         "params": config.XGBoostPrams.params,
        #         "train_params": config.XGBoostPrams.train_params,
        #     },
        # )
        # writer.log_metric("XGB_Accuracy", xgb_accuracy)
        # writer.log_artifact("../data/working/xgb_models.pkl")
        # writer.log_figure(xgb_importance_fig, "xgb_importance.png")
        # xgb_importance_fig.savefig("../data/working/xgb_importance.png")
        # # Ensemble
        # writer.log_metric("Ensemble_Accuracy", ensemble_accuracy)
        # # Close writer client.
        # writer.set_terminated()


if __name__ == "__main__":
    with timer("Time of Train Processing"):
        main()
