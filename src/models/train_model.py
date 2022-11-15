import os
import gc

import gcsfs
from optuna.integration.wandb import WeightsAndBiasesCallback
import optuna
import pandas as pd

import wandb

from src.models.objective import ClassicalObjective, set_seed


def init_gcloud() -> None:
    # see start.sh for location
    gcloud_config = os.path.abspath(
        os.path.expanduser(
            os.path.expandvars("~/.config/gcloud/application_default_credentials.json")
        )
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_config
    os.environ["GCLOUD_PROJECT"] = "flowing-mantis-239216"
    gcsfs.GCSFileSystem(project="thesis", token=gcloud_config)


if __name__ == "__main__":

    init_gcloud()

    # init new 
    run = wandb.init(project="thesis", entity="fbv", name="ClassicalClassifier")
    artifact = run.use_artifact("train_val_test:v0")
    artifact_dir = artifact.download()

    # FIXME: Change later as needed.
    val = pd.read_parquet("artifacts/train_val_test:v0/data_preprocessed_2017")
    x_train = val.sample(n=10)
    x_val = x_train
    y_train = x_train["buy_sell"]
    y_val = y_train

    wandb_kwargs = {"project": "thesis"}
    wandbc = WeightsAndBiasesCallback(
        metric_name="accuracy",
        wandb_kwargs={"project": "thesis"},
    )

    # maximize for accuracy
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=set_seed()),
        study_name="study_classical_classifier",
    )

    # run garbage collector after each trial. Might impact performance,
    # but can mitigate out-of-memory errors.
    study.optimize(
        ClassicalObjective(x_train, y_train, x_val, y_val),
        n_trials=100,
        timeout=600,
        callbacks=[lambda study, trial: gc.collect(), wandbc],
    )

  
    print(f"best trial: {study.best_trial.number}")
    print(f"params: {study.best_params}")
    print(f"value: {study.best_value}")

    run.finish()
