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
from optuna.exceptions import ExperimentalWarning
from optuna.integration.wandb import WeightsAndBiasesCallback

from otc.config.config import settings
from otc.features.build_features import (
    features_categorical,
    features_classical,
    features_classical_size,
    features_ml,
)
from otc.models.objective import (
    ClassicalObjective,
    FTTransformerObjective,
    GradientBoostingObjective,
    TabNetObjective,
    TabTransformerObjective,
    set_seed,
)

OBJECTIVES = {
    "gbm": GradientBoostingObjective,
    "classical": ClassicalObjective,
    "tabtransformer": TabTransformerObjective,
    "fttransformer": FTTransformerObjective,
    "tabnet": TabNetObjective,
}

FEATURE_SETS = {
    "classical": features_classical,
    "ml": features_ml,
    "classical-size": features_classical_size,
}


@click.command()
@click.option("--trials", default=100, help="No. of trials")
@click.option("--seed", default=42, required=False, type=int, help="Seed for rng.")
@click.option(
    "--features",
    type=click.Choice(["classical", "ml", "classical-size"], case_sensitive=False),
    default="classical",
    help="Feature set to run study on.",
)
@click.option(
    "--model",
    type=click.Choice(
        ["classical", "gbm", "tabtransformer", "fttransformer", "tabnet"],
        case_sensitive=False,
    ),
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
def main(
    trials: int,
    seed: int,
    features: str,
    model: str,
    name: str,
    dataset: str,
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
    """
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    logger.info("Connecting to weights & biases. Downloading artifacts. 📦")

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

    # select right feature set
    columns = ["buy_sell"]
    columns.extend(FEATURE_SETS[features])

    # filter categorical features that are in subset and get cardinality
    cat_features_sub = [tup for tup in features_categorical if tup[0] in columns]

    cat_features, cat_cardinalities = [], []
    if cat_features_sub:
        cat_features, cat_cardinalities = tuple(list(t) for t in zip(*cat_features_sub))

    # load data
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

    # select right objective based on model nome, constructor might be overloaded
    objective = OBJECTIVES[model](
        x_train,
        y_train,
        x_val,
        y_val,
        cat_features=cat_features,
        cat_cardinalities=cat_cardinalities,
    )

    # maximize for accuracy
    study = optuna.create_study(
        direction="maximize",
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=set_seed(seed)),
        study_name=name,
    )

    # run garbage collector after each trial. Might impact performance,
    # but can mitigate out-of-memory errors.
    # Save models in `objective_callback`.
    study.optimize(
        objective,
        n_trials=trials,
        gc_after_trial=True,
        callbacks=[wand_cb, objective.objective_callback],
        show_progress_bar=True,
    )

    logger.info("writing artifacts to weights and biases. 🗃️")

    # provide summary statistics
    wandb.run.summary.update(  # type: ignore
        {
            "best_accuracy": study.best_trial.value,
            "best_trial": study.best_trial.number,
            "features": features,
            "trials": trials,
            "name": name,
            "dataset": dataset,
            "seed": seed,
        }
    )

    wandb.log(  # type: ignore
        {
            "plot_optimization_history": optuna.visualization.plot_optimization_history(
                study
            ),
            "plot_param_importances": optuna.visualization.plot_param_importances(
                study
            ),
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
