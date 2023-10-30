"""Holds configuration for folders, dbs, and wandb configuration.

See also `prod.env`.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Specifies settings.

    Mainly W&B, GCS and Heroku.
    """

    WANDB_PROJECT: str
    WANDB_ENTITY: str

    GCS_PROJECT_ID: str
    GCS_CRED_FILE: Path
    GCS_BUCKET: str

    MODEL_DIR_REMOTE: Path

    class Config:
        """Specifies configuration.

        Filename is given by "prod.env". Keys are case-sensitive.
        """

        case_sensitive = True
        env_file = "prod.env"
        env_file_encoding = "utf-8"


settings = Settings(_env_file="prod.env", _env_file_encoding="utf-8")
