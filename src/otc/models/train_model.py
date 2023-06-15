"""
Script to perform a hyperparameter search for various models.

Currently classical rules and gradient boosted trees are supported.

"""
import logging
import logging.config
import pickle
import sys
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
    set_seed,
)

OBJECTIVES = {
    "gbm": GradientBoostingObjective,
    "classical": ClassicalObjective,
    "fttransformer": FTTransformerObjective,
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
        ["classical", "gbm", "fttransformer"],
        case_sensitive=False,
    ),
    required=True,
    default="classical-size",
    help="Feature set to run study on.",
)
@click.option("--id", required=False, type=str, help="Id of run / name of study.")
@click.option(
    "--dataset",
    required=False,
    default="fbv/thesis/ise_supervised_none:latest",
    help="Name of dataset. See W&B Artifacts/Full Name",
)
@click.option(
    "--pretrain/--no-pretrain", default=False, help="Flag to activate pretraining."
)
@click.option(
    "--sample",
    type=click.FloatRange(0, 1),
    default=1,
    help="Sampling factor applied to train and validation set.",
)
def main(
    trials: int,
    seed: int,
    features: str,
    model: str,
    id: str,
    dataset: str,
    pretrain: bool,
) -> None:
    """
    Start study.

    Args:
        trials (int): no. of trials.
        seed (int): seed for rng.
        features (str): name of feature set.
        model (str): name of model.
        id (str): id of study.
        dataset (str): name of data set.
        pretrain (bool): whether to pretrain model.
    """
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    logger.info("Connecting to weights & biases. Downloading artifacts. 📦")

    run = wandb.init(  # type: ignore
        project=settings.WANDB_PROJECT, entity=settings.WANDB_ENTITY, name=id
    )

    if not id:
        # replace missing names with run id and create new sampler
        id = str(run.id)
        sampler = optuna.samplers.TPESampler(seed=set_seed(seed))
    else:
        # download saved study
        artifact_study = run.use_artifact(id + ".optuna:latest")
        artifact_dir = artifact_study.download()

        saved_study = pickle.load(open(Path(artifact_dir, id + ".optuna"), "rb"))
        sampler = saved_study.sampler

    # select right feature set
    columns = ["buy_sell"]
    columns.extend(FEATURE_SETS[features])

    # filter categorical features that are in subset and get cardinality
    cat_features_sub = [tup for tup in features_categorical if tup[0] in columns]

    cat_features, cat_cardinalities = [], []
    if cat_features_sub:
        cat_features, cat_cardinalities = tuple(list(t) for t in zip(*cat_features_sub))

    logger.info("Start loading artifacts locally. 🐢")

    # supervised data
    artifact_labelled = run.use_artifact(dataset)
    artifact_dir_labelled = artifact_labelled.download()

    # Load labelled data
    x_train = pd.read_parquet(
        Path(artifact_dir_labelled, "train_set.parquet"), columns=columns
    )
    y_train = x_train["buy_sell"]
    x_train.drop(columns=["buy_sell"], inplace=True)

    if pretrain:
        # Load unlabelled data
        unlabelled_dataset = dataset.replace("supervised", "unsupervised")
        artifact_unlabelled = run.use_artifact(unlabelled_dataset)
        artifact_dir_unlabelled = artifact_unlabelled.download()
        x_train_unlabelled = pd.read_parquet(
            Path(artifact_dir_unlabelled, "train_set.parquet"), columns=columns
        )
        y_train_unlabelled = x_train_unlabelled["buy_sell"]
        x_train_unlabelled.drop(columns=["buy_sell"], inplace=True)

        # Concatenate labelled and unlabelled data unlabelled will merge in between
        x_train = pd.concat([x_train, x_train_unlabelled])
        y_train = pd.concat([y_train, y_train_unlabelled])

    # load validation data
    x_val = pd.read_parquet(
        Path(artifact_dir_labelled, "val_set.parquet"), columns=columns
    )
    y_val = x_val["buy_sell"]
    x_val.drop(columns=["buy_sell"], inplace=True)

    # pretrain training activated
    has_label = (y_train != 0).all()
    if pretrain and has_label:
        raise ValueError(
            "Pretraining active, but dataset contains no unlabelled instances."
        )

    # no pretraining activated
    has_label = y_train.isin([-1, 1]).all()
    if not pretrain and not has_label:
        raise ValueError(
            "Pretraining inactive, but dataset contains unlabelled instances or"
            "other labels. Use different dataset or activate pretraining."
        )

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
        pretrain=pretrain,
    )

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # create study.
    study = optuna.create_study(
        sampler=sampler,
        study_name=id,  # match id of run and study
        storage=f"sqlite:///{id}.db",
        load_if_exists=True,
        direction="maximize",  # maximize for accuracy
    )

    # Save models in `objective_callback`.
    study.optimize(
        objective,
        n_trials=trials,
        # timeout=60 * 60 * 12,  # 12 hours
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
            "name": id,
            "dataset": dataset,
            "seed": seed,
            "pretrain": pretrain,
            "sample": 1.0,
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
