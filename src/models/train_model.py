import os

import gcsfs
import optuna
import pandas as pd

import wandb


def init_gcloud()->None:
    # see start.sh for location
    gcloud_config = os.path.abspath(
        os.path.expanduser(
            os.path.expandvars("~/.config/gcloud/application_default_credentials.json")
        )
    )
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcloud_config
    os.environ['GCLOUD_PROJECT'] = "flowing-mantis-239216"
    gcsfs.GCSFileSystem(project="thesis", token=gcloud_config)

if __name__ == "__main__":

    init_gcloud()

    run = wandb.init(project="thesis",entity="fbv")
    artifact = run.use_artifact("train_val_test:v0")
    artifact_dir = artifact.download()

    val = pd.read_parquet("artifacts/train_val_test:v0/val_set_20")

    print(val.head())