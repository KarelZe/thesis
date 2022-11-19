"""
Script to perform a hyperparameter search for various models.

Currently classical rules and gradient boosted trees are supported.

"""
import logging
import logging.config
import warnings
from pathlib import Path

import click
import optuna
import pandas as pd
import wandb
import yaml
from features.build_features import (
    features_categorical,
    features_classical,
    features_ml,
)
from optuna.exceptions import ExperimentalWarning
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import RetryFailedTrialCallback
from otc.utils.config import Settings

from otc.models.objective import (
    ClassicalObjective,
    GradientBoostingObjective,
    Objective,
    set_seed,
)


@click.command()
@click.option("--trials", default=100, help="No. of trials")
@click.option("--seed", default=42, required=False, type=int, help="Seed for rng.")
@click.option(
    "--features",
    type=click.Choice(["classical", "ml"], case_sensitive=False),
    default="classical",
    help="Feature set to run study on.",
)
@click.option(
    "--model",
    type=click.Choice(["classical", "gbm"], case_sensitive=False),
    required=True,
    default="classical",
    help="Feature set to run study on.",
)
@click.option("--name", required=False, type=str, help="Name of study.")
@click.option(
    "--dataset",
    required=False,
    default="fbv/thesis/train_val_test:v0",
    help="Name of dataset. See W&B Artifacts/Full Name",
)
@click.option(
    "--mode",
    type=click.Choice(["resume", "start"], case_sensitive=False),
    default="start",
    help="Start a new study or resume an existing study with given name.",
)
def main(
    trials: int,
    seed: int,
    features: str,
    model: str,
    name: str,
    dataset: str,
    mode: str,
) -> None:
    """
    Start study.

    Args:
        trials (int): no. of trials.
        seed (int): seed for rng.
        features (str): name of feature set.
        model (str): name of model.
        name (str): name of study.
        dataset (str): name of data set.
        mode (str): mode to run study on.
    """
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    logger.info("Connecting to weights & biases. Downloading artifacts. 📦")

    settings = Settings()

    run = wandb.init(  # type: ignore
        project=settings.WANDB_PROJECT,
        entity=settings.WANDB_ENTITY,
        name=name,
    )

    # replace missing names with run id
    if not name:
        name = str(run.id)

    artifact = run.use_artifact(dataset)
    artifact_dir = artifact.download()

    logger.info("Start loading artifacts locally. 🐢")

    columns = ["buy_sell"]
    if features == "classical":
        columns.extend(features_classical)
    elif features == "ml":
        columns.extend(features_ml)

    x_train = pd.read_parquet(
        Path(artifact_dir, "train_set_60.parquet"), columns=columns
    )
    y_train = x_train["buy_sell"]
    x_train.drop(columns=["buy_sell"], inplace=True)

    x_val = pd.read_parquet(Path(artifact_dir, "val_set_20.parquet"), columns=columns)
    y_val = x_val["buy_sell"]
    x_val.drop(columns=["buy_sell"], inplace=True)

    wand_cb = WeightsAndBiasesCallback(
        metric_name="accuracy",
        wandb_kwargs={"project": settings.WANDB_PROJECT},
    )

    logger.info("Start with study. 🦄")

    objective: Objective
    if model == "gbm":
        objective = GradientBoostingObjective(
            x_train,
            y_train,
            x_val,
            y_val,
            cat_features=features_categorical,
        )
    elif model == "classical":
        objective = ClassicalObjective(x_train, y_train, x_val, y_val)

    storage = optuna.storages.RDBStorage(
        url=settings.OPTUNA_RDB,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )

    # maximize for accuracy
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=set_seed(seed)),
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        study_name=name,
        storage=storage,
        load_if_exists=bool(mode == "resume"),
    )

    # run garbage collector after each trial. Might impact performance,
    # but can mitigate out-of-memory errors.
    # Save models using objective.save_callback
    study.optimize(
        objective,
        n_trials=trials,
        gc_after_trial=True,
        callbacks=[wand_cb, objective.save_callback],
        show_progress_bar=True,
    )

    logger.info("writing artifacts to weights and biases. 🗃️")

    wandb.run.summary["best accuracy"] = study.best_trial.value  # type: ignore
    wandb.run.summary["best trial"] = study.best_trial.number  # type: ignore
    wandb.run.summary["features"] = features  # type: ignore
    wandb.run.summary["trials"] = trials  # type: ignore
    wandb.run.summary["name"] = name  # type: ignore
    wandb.run.summary["dataset"] = dataset  # type: ignore
    wandb.run.summary["seed"] = seed  # type: ignore
    wandb.run.summary["mode"] = mode  # type: ignore

    wandb.log(  # type: ignore
        {
            "optimization_history": optuna.visualization.plot_optimization_history(
                study
            ),
            "param_importances": optuna.visualization.plot_param_importances(study),
            "plot_contour": optuna.visualization.plot_contour(study),
        }
    )

    run.finish()

    logger.info("All done! ✨ 🍰 ✨")


if __name__ == "__main__":

    with open("logging.yaml") as file:
        loaded_config = yaml.safe_load(file)
        logging.config.dictConfig(loaded_config)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
