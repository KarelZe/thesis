import os

import gcsfs


def _create_environment() -> gcsfs.GCSFileSystem:
    """
    Implementation of the global object pattern to connect only once to GCS.

    See https://python-patterns.guide/python/module-globals/
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


fs = _create_environment()

# class Singleton(gcsfs.GCSFileSystem):
#     _instances = {}

#     def __new__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
#         return cls._instances[cls]


# class GCS(Singleton):
#     def __init__(self):
#         # see start.sh for location
#         gcloud_config = os.path.abspath(
#             os.path.expanduser(
#                 os.path.expandvars(
#                     "~/.config/gcloud/application_default_credentials.json"
#                 )
#             )
#         )
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_config
#         os.environ["GCLOUD_PROJECT"] = "flowing-mantis-239216"
#         gcsfs.GCSFileSystem(project="thesis", token=gcloud_config)
