
from __future__ import annotations

import logging
import logging.config
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def on_epoch_end(self, epoch:int, epochs:int, train_loss:float, val_loss:float):
        pass

    def on_train_end(
        self, study: optuna.Study, trial: optuna.Trial, model: Any, name: str
    ):
        pass


class SaveCallback(Callback):
    def __init__(self, wandb_kwargs: Optional[Dict[str, Any]] = None):
        self._wandb_kwargs = wandb_kwargs or {}

        # create wandb run if it doesn't exist
        self._run = wandb.run
        if not self._run:
            self._run = self._initialize_run()

    def _initialize_run(self) -> "wandb.sdk.wandb_run.Run":
        """Initializes Weights & Biases run."""
        run = wandb.init(**self._wandb_kwargs)
        if not isinstance(run, wandb.sdk.wandb_run.Run):
            raise RuntimeError(
                "Cannot create a Run. "
                "Expected wandb.sdk.wandb_run.Run as a return. "
                f"Got: {type(run)}."
            )
        return run

    def on_train_end(
        self, study: optuna.Study, trial: optuna.Trial, model: nn.Module | CatBoostClassifier, name: str
    ):

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
                # https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
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
            model_artifact = wandb.Artifact(name=new_file, type="model")
            model_artifact.add_reference(remote_path, name=new_file)
            self._run.log_artifact(model_artifact)
            logger.info("%sSaved '%s'.%s", Colors.OKGREEN, new_file, Colors.ENDC)


class PrintCallback(Callback):
    def on_epoch_end(self, epoch, epochs, train_loss, val_loss):

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
    """

    callbacks: List[Callback] = field(default_factory=list)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_end(self, epoch, epochs, train_loss, val_loss):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, epochs, train_loss, val_loss)

    def on_train_end(
        self, study: optuna.Study, trial: optuna.Trial, model: Any, name: str
    ):
        for callback in self.callbacks:
            callback.on_train_end(study, trial, model, name)
