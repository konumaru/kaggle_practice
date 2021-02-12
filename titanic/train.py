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
from mikasa.trainer.base import SklearnRegressionTrainer
from mikasa.trainer.cross_validation import RSACrossValidationTrainer
from mikasa.ensemble import SimpleAgerageEnsember

from mikasa.plot import plot_importance
from mikasa.mlflow_writer import MlflowWriter


def run_train(model_name, base_trainer, X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
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


def eval_single_model(trainer, data, target):
    pred = np.array(trainer.predict(data)).T

    ensembler = SimpleAgerageEnsember()
    pred_avg = ensembler.predict(pred)
    pred_avg = np.where(pred_avg > 0.5, 1, 0)
    score = accuracy_score(target, pred_avg)
    return score


def save_importance(model_name, trainer):
    name, mean_importance, std_importance = trainer.get_importance()
    fig = plot_importance(name, mean_importance, std_importance)
    fig.savefig(f"../data/titanic/working/{model_name}_importance.png")


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

    cv_metrics = {}
    eval_metrics = {}

    # Train model.
    base_trainers = {
        "LGBM": LGBMTrainer(
            config.LightgbmParams.params, config.LightgbmParams.train_params
        ),
    }
    fit_trainers = {}
    for model_name, base_trainer in base_trainers.items():
        trainer, metric = run_train(model_name, base_trainer, X_train, y_train)

        fit_trainers[model_name] = trainer
        cv_metrics[model_name] = metric
        eval_metrics[model_name] = eval_single_model(trainer, X_eval, y_eval)
        save_importance(model_name, trainer)

    # Stacking
    pred_first = []
    for model_name, _trainer in fit_trainers.items():
        pred_first.append(np.array(_trainer.predict(X_eval)).T)
    pred_first = np.concatenate(pred_first, axis=1)
    pred_first = pd.DataFrame(pred_first)

    base_trainer = SklearnRegressionTrainer(model=Ridge(random_state=config.SEED))
    trainer, metric = run_train("stack_ridge", base_trainer, pred_first, y_eval)
    eval_metrics["Stack"] = metric

    # Evaluation
    for model_name, metric in cv_metrics.items():
        print(f"{model_name:>8} CV   Metric: {metric:.08f}")
    for model_name, metric in eval_metrics.items():
        print(f"{model_name:>8} Eval Metric: {metric:.08f}")

    # Domp logs to mlflow.
    if config.DEBUG is not True:
        writer = MlflowWriter(
            config.MLflowConfig.experiment_name,
            tracking_uri=os.path.abspath("../mlruns"),
        )
        writer.set_run_name(config.MLflowConfig.run_name)
        writer.set_note_content(config.MLflowConfig.experiment_note)
        # Features
        writer.log_param("Feature", ", ".join(config.FeatureList.features))
        # Paraeters
        writer.log_param("SEED", config.SEED)
        writer.log_param("NUM_SEED", config.NUM_SEED)
        writer.log_param(
            "LGBM_params",
            {
                "params": config.LightgbmParams.params,
                "train_params": config.LightgbmParams.train_params,
            },
        )
        # Metric
        for model_name, _metric in cv_metrics.items():
            writer.log_metric(f"{model_name} CV Metric", _metric)
        for model_name, _metric in eval_metrics.items():
            writer.log_metric(f"{model_name} Eval Metric", _metric)
        # Close writer client.
        writer.set_terminated()


if __name__ == "__main__":
    with timer("Train Processing"):
        main()
