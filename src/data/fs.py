"""
Gives simple access to the google cloud storage bucket.

Instance is only created once.
"""

import os

import gcsfs

from src.data import const


def _create_environment() -> gcsfs.GCSFileSystem:
    """
    Implement the global object pattern to connect only once to GCS.

    Returns:
        gcsfs.GCSFileSystem: Instance of GCSFileSystem.
    """
    gcloud_config = os.path.abspath(
        os.path.expanduser(os.path.expandvars(const.GCS_CRED_FILE))
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_config
    os.environ["GCLOUD_PROJECT"] = const.GCS_PROJECT_ID
    return gcsfs.GCSFileSystem(project=const.GCS_PROJECT_ID, token=gcloud_config)


# global object pattern. See https://python-patterns.guide/python/module-globals/
fs = _create_environment()
