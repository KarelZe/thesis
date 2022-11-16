import warnings

import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import RetryFailedTrialCallback

import wandb

from src.models.objective import GradientBoostingObjective, set_seed
from src.data.fs import fs


if __name__ == "__main__":

    # init new
    run = wandb.init(project="thesis", entity="fbv", name="GradientBoostedTrees")
    artifact = run.use_artifact("train_val_test:v0")
    artifact_dir = artifact.download()

    # FIXME: Change later as needed.
    val = pd.read_parquet("artifacts/train_val_test:v0/data_preprocessed_2017")
    x_train = val[["bid_ex","ask_ex", "buy_sell"]].sample(n=10)
    x_val = x_train
    y_train = x_train["buy_sell"]
    y_val = y_train

    wandb_kwargs = {"project": "thesis"}
    wandbc = WeightsAndBiasesCallback(
        metric_name="accuracy",
        wandb_kwargs={"project": "thesis"},
    )

    warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.")

    storage = optuna.storages.RDBStorage(
        url="postgresql://vvtzcgrpjuvzro:4454a77e98a3cb825d"
        "080f0161b082e5101d401cd1abb922e673e93e7321b4bf@ec2-52-49-120-150"
        ".eu-west-1.compute.amazonaws.com:5432/d4d1dtdcorfq8",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )

    # maximize for accuracy
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=set_seed()),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        study_name=f"{run.id}",
        storage=storage,
    )

    # run garbage collector after each trial. Might impact performance,
    # but can mitigate out-of-memory errors.
    objective = GradientBoostingObjective(x_train, y_train, x_val, y_val, features=x_val.columns.tolist())
    study.optimize(
        objective,
        n_trials=10,
        timeout=600,
        gc_after_trial=True,
        callbacks=[wandbc, objective.save_callback],
        show_progress_bar=True
    )


    # fs.put("./models/","gs://thesis-bucket-option-trade-classification/models/", recursive=True)
    

    # model_artifact = wandb.Artifact('gradient-boosted-tree', type='model')
    # model_artifact.add_reference('gs://thesis-bucket-option-trade-classification/models/gbt/')
    # run.log_artifact(model_artifact)

    print(f"best trial: {study.best_trial.number}")
    print(f"params: {study.best_params}")
    print(f"value: {study.best_value}")

    # study.optimize(
    #     ClassicalObjective(x_train, y_train, x_val, y_val),
    #     n_trials=100,
    #     timeout=600,
    #     callbacks=[lambda study, trial: gc.collect(), wandbc],
    # )

    run.finish()
