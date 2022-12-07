"""
Gives simple access to the google cloud storage bucket.

Instance is only created once.
"""

import os
from pathlib import Path

import gcsfs

from otc.utils.config import settings


def _create_environment() -> gcsfs.GCSFileSystem:
    """
    Implement the global object pattern to connect only once to GCS.

    Returns:
        gcsfs.GCSFileSystem: Instance of GCSFileSystem.
    """
    gcloud_config = str(Path(settings.GCS_CRED_FILE).expanduser().resolve())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_config
    os.environ["GCLOUD_PROJECT"] = settings.GCS_PROJECT_ID
    return gcsfs.GCSFileSystem(project=settings.GCS_PROJECT_ID, token=gcloud_config)


# global object pattern. See https://python-patterns.guide/python/module-globals/
fs = _create_environment()
