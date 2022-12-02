"""
Implementation of callbacks for neural nets and other models.

TODO: Refactor early stoppping to callback.
"""

from __future__ import annotations

import logging
import logging.config
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import torch
from catboost import CatBoostClassifier
from torch import nn

import wandb
from otc.data.fs import fs
from otc.utils.colors import Colors
from otc.utils.config import settings

logger = logging.getLogger(__name__)


class Callback:
    """
    Abstract base class used to build new callbacks.

    Concrete Callbacks must implement some of the methods.
    """

    def __init__(self) -> None:
        """
        Initialize the callback.

        May be overwritten in subclass.
        """

    def set_params(self, params: Any) -> None:
        """
        Set the parameters of the callback.

        Args:
            params (Any): params.
        """
        self.params = params

    def on_epoch_end(
        self, epoch: int, epochs: int, train_loss: float, val_loss: float
    ) -> None:
        """
        Call at the end of each epoch.

        Args:
            epoch (int): current epoch.
            epochs (int): total number of epochs.
            train_loss (float): train loss in epoch.
            val_loss (float): validation loss in epoch.
        """

    def on_train_end(
        self, study: optuna.Study, trial: optuna.Trial, model: Any, name: str
    ) -> None:
        """
        Call on_train_end for each callback in container.

        Args:
            study (optuna.Study): optuna study.
            trial (optuna.Trial): optuna trial.
            model (nn.Module | CatBoostClassifier): model.
            name (str): name of study.
        """


class SaveCallback(Callback):
    """
    Callback to save the models.

    Args:
        Callback (_type_): _description_
    """

    def __init__(self, wandb_kwargs: dict[str, Any] | None = None) -> None:
        """
        Initialize the callback.

        Similar to optuna wandb callback, but with the ability to save models to GCS.
        See: https://bit.ly/3OSGFyU

        Args:
            wandb_kwargs (dict[str, Any] | None, optional): kwargs of wandb.
            Defaults to None.
        """
        self._wandb_kwargs = wandb_kwargs or {}  # type: ignore
        self._run = wandb.run  # type: ignore
        if not self._run:
            self._run = self._initialize_run()

    def _initialize_run(self) -> wandb.sdk.wandb_run.Run:  # type: ignore
        """
        Initialize wandb run.

        Adapted from: https://bit.ly/3OSGFyU.
        """
        run = wandb.init(**self._wandb_kwargs)  # type: ignore
        if not isinstance(run, wandb.sdk.wandb_run.Run):  # type: ignore
            raise RuntimeError(
                "Cannot create a Run. "
                "Expected wandb.sdk.wandb_run.Run as a return."
                f"Got: {type(run)}."
            )
        return run

    def on_train_end(
        self,
        study: optuna.Study,
        trial: optuna.Trial,
        model: nn.Module | CatBoostClassifier,
        name: str,
    ) -> None:
        """
        Save the model at the end of the training, if it is the best model in the study.

        Delete old models from GCS from previous trials of the same study. References to
        the old models are logged in wandb.

        For CatBoostClassifier, save the model as a pickle file.
        For PyTorch models, save the model as a state_dict.

        Args:
            study (optuna.Study): optuna study.
            trial (optuna.Trial): optuna trial.
            model (nn.Module | CatBoostClassifier): model.
            name (str): name of study.
        """
        if study.best_trial == trial:

            prefix_file = (
                f"{study.study_name}_" f"{model.__class__.__name__}_{name}_trial_"
            )

            # remove old files on remote
            outdated_files_remote = fs.glob(
                "gs://"
                + Path(
                    settings.GCS_BUCKET, settings.MODEL_DIR_REMOTE, prefix_file + "*"
                ).as_posix()
            )
            if len(outdated_files_remote) > 0:
                fs.rm(outdated_files_remote)
                logger.info(
                    "%sRemoved %s.%s",
                    Colors.FAIL,
                    outdated_files_remote,
                    Colors.ENDC,
                )

            remote_path: str
            new_file: str

            # write new files on remote
            if isinstance(model, CatBoostClassifier):
                # https://catboost.ai/en/docs/concepts/python\
                # -reference_catboost_save_model
                new_file = prefix_file + f"{trial.number}.cbm"
                remote_path = (
                    "gs://"
                    + Path(
                        settings.GCS_BUCKET, settings.MODEL_DIR_REMOTE, new_file
                    ).as_posix()
                )
                with fs.open(remote_path, "wb") as f:
                    f.write(model._serialize_model())
            elif isinstance(model, nn.Module):
                new_file = prefix_file + f"{trial.number}.pth"
                remote_path = (
                    "gs://"
                    + Path(
                        settings.GCS_BUCKET, settings.MODEL_DIR_REMOTE, new_file
                    ).as_posix()
                )
                # https://stackoverflow.com/a/72511896/5755604
                with fs.open(remote_path, "wb") as f:
                    torch.save(model.state_dict(), f)
            else:
                return

            # log to wandb
            model_artifact = wandb.Artifact(name=new_file, type="model")  # type: ignore
            model_artifact.add_reference(remote_path, name=new_file)
            self._run.log_artifact(model_artifact)  # type: ignore
            logger.info("%sSaved '%s'.%s", Colors.OKGREEN, new_file, Colors.ENDC)


class PrintCallback(Callback):
    """
    Callback to print train and validation loss.

    Args:
        Callback (callback): callback.
    """

    def on_epoch_end(
        self, epoch: int, epochs: int, train_loss: float, val_loss: float
    ) -> None:
        """
        Print train and validation loss on each epoch.

        Args:
            epoch (int): current epoch.
            epochs (int): total number of epochs.
            train_loss (float): train loss in epoch.
            val_loss (float): validation loss in epoch.
        """
        logger.info(
            "%s[epoch %04d/%04d]%s %strain loss:%s %.8f %sval loss:%s %.8f",
            Colors.OKGREEN,
            epoch + 1,
            epochs,
            Colors.ENDC,
            Colors.BOLD,
            Colors.ENDC,
            train_loss,
            Colors.BOLD,
            Colors.ENDC,
            val_loss,
        )


@dataclass
class CallbackContainer:
    """
    Container holding a list of callbacks.

    Register using append method.
    """

    callbacks: list[Callback] = field(default_factory=list)

    def append(self, callback: Callback) -> None:
        """
        Add a callback to the container.

        Args:
            callback (Callback): callback to add.
        """
        self.callbacks.append(callback)

    def set_params(self, params: Any) -> None:
        """
        Set params for callbacks in container.

        Args:
            params (Any): parameter.
        """
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_end(
        self, epoch: int, epochs: int, train_loss: float, val_loss: float
    ) -> None:
        """
        Call on_epoch_end for each callback in container.

        Args:
            epoch (int): current epoch.
            epochs (int): total number of epochs.
            train_loss (float): train loss in epoch.
            val_loss (float): validation loss in epoch.
        """
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, epochs, train_loss, val_loss)

    def on_train_end(
        self,
        study: optuna.Study,
        trial: optuna.Trial,
        model: nn.Module | CatBoostClassifier,
        name: str,
    ) -> None:
        """
        Call on_train_end for each callback in container.

        Args:
            study (optuna.Study): optuna study.
            trial (optuna.Trial): optuna trial.
            model (nn.Module | CatBoostClassifier): model.
            name (str): name of study.
        """
        for callback in self.callbacks:
            callback.on_train_end(study, trial, model, name)
