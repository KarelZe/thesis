import logging
import warnings
from pathlib import Path

import click
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import RetryFailedTrialCallback

import wandb
from src.models.objective import ClassicalObjective, GradientBoostingObjective, set_seed

@click.command()
@click.option("--trials", default=100, help="No. of trials")
@click.option("--seed", default=42, required=False, type=int, help="Seed for rng.")
@click.option(
    "--features",
    type=click.Choice(["fs1", "fs2", "fs3", "fs4"], case_sensitive=False),
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
    "--mode",
    type=click.Choice(["resume", "start"], case_sensitive=False),
    default="start",
    help="Start a new study or resume an existing study with given name.",
)
def main(
    trials: int, seed: int, features: str, model: str, name: str, mode: str
) -> None:  # pragma: no cover
    """
    Start study.

    Args:
        trials (int): no. of trials.
        seed (int): seed for rng.
        features (str): name of feature set.
        model (str): name of model.
        name (str): name of study.
        mode (str): mode to run study on.
    """
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    logger.info("Connecting to weights & biases. Downloading artifacts. 📦")
    run = wandb.init(project="thesis", entity="fbv", name="GradientBoostedTrees")
    artifact = run.use_artifact("train_val_test:v0")
    artifact_dir = artifact.download()

    logger.info("Start loading artifacts locally. 🐢")
    # FIXME: Change later as needed. Filter later if features are known.
    val = pd.read_parquet(artifact_dir + "/data_preprocessed_2017")
    x_train = val[[ 'TRADE_SIZE', 'TRADE_PRICE',
       'BEST_BID', 'BEST_ASK', 'order_id', 'ask_ex', 'bid_ex', 'bid_size_ex',
       'ask_size_ex', 'price_all_lead', 'price_all_lag', 'optionid', 'day_vol',
       'price_ex_lead', 'price_ex_lag', 'buy_sell']].sample(n=10)
    x_val = x_train
    y_train = x_train["buy_sell"]
    y_val = y_train

    wandbc = WeightsAndBiasesCallback(
        metric_name="accuracy",
        wandb_kwargs={"project": "thesis"},
    )

    logger.info("Start with study. 🦄")

    objective = None
    if model == "gbm":
        objective = GradientBoostingObjective(
            x_train, y_train, x_val, y_val, features=x_val.columns.tolist()
        )
    elif model == "classical":
        objective = ClassicalObjective(x_train, y_train, x_val, y_val)

    # replace missing names with run id
    if not name:
        name = str(run.id)

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
        timeout=600,
        gc_after_trial=True,
        callbacks=[wandbc, objective.save_callback],
        show_progress_bar=True,
    )

    logger.info("writing artifacts to weights and biases. 🗃️")
    wandb.run.summary["best accuracy"] = study.best_trial.value
    wandb.run.summary["best trial"] = study.best_trial.number

    wandb.log(
        {
            "optuna_optimization_history": optuna.visualization.plot_optimization_history(
                study
            ),
            # "optuna_param_importances": optuna.visualization.plot_param_importances(
            #     study
            # ),
            "optuna_plot_contour": optuna.visualization.plot_contour(
                study,
            ),
        }
    )

    run.finish()

    logger.info("All done! ✨ 🍰 ✨")


if __name__ == "__main__":

    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
