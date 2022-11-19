from typing import Final
from pydantic import BaseSettings, PostgresDsn
from pathlib import Path

class Settings(BaseSettings):
    WANDB_PROJECT: Final[str]
    WANDB_ENTITY: Final[str]
    OPTUNA_RDB: Final[PostgresDsn]

    GCS_PROJECT_ID: Final[str]
    GCS_CRED_FILE: Final[Path]
    GCS_BUCKET: Final[str]

    MODEL_DIR_LOCAL: Final[Path]
    MODEL_DIR_REMOTE: Final[Path]
    
    class Config:
        case_sensitive = True
        env_file = "prod.env"
        env_file_encoding = "utf-8"

