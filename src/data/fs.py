"""
Gives simple access to the google cloud storage bucket.

Instance is only created once.
"""

import os

import gcsfs


def _create_environment() -> gcsfs.GCSFileSystem:
    """
    Implement the global object pattern to connect only once to GCS.

    Returns:
        gcsfs.GCSFileSystem: Instance of GCSFileSystem.
    """
    # see start.sh for location
    gcloud_config = os.path.abspath(
        os.path.expanduser(
            os.path.expandvars("~/.config/gcloud/application_default_credentials.json")
        )
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_config
    os.environ["GCLOUD_PROJECT"] = "flowing-mantis-239216"
    return gcsfs.GCSFileSystem(project="thesis", token=gcloud_config)


# global object pattern. See https://python-patterns.guide/python/module-globals/
fs = _create_environment()
