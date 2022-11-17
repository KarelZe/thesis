"""
Provides a welcoming screen and consts used in the scripts.

It contains the name of the chair in stylized ASCII.
"""

from typing import Final

WELCOME = """
,dPYb,,dPYb,
 IP'`YbIP'`Yb
 I8  8II8  8I
 I8  8'I8  8'
 I8 dP I8 dP        ggg    gg
 I8dP  I8dP   88gg d8"Yb   88bg
 I8P   I8P    8I  dP  I8   8I
,d8b,_,d8b,  ,8I,dP   I8, ,8I
PI8"888P'"Y88P"'8"     "Y8P"
 I8 `8,
 I8  `8,
 I8   8I
 I8   8I
 I8, ,8'
  "Y8P'
"""

WANDB_PROJECT: Final[str] = "thesis"
WANDB_ENTITY: Final[str] = "fbv"

OPTUNA_RDB: Final[str] = (
    "postgresql://vvtzcgrpjuvzro:4454a77e98a3cb825d"
    "080f0161b082e5101d401cd1abb922e673e93e7321b4bf@ec2-52-49-120-150"
    ".eu-west-1.compute.amazonaws.com:5432/d4d1dtdcorfq8"
)

GCS_PROJECT_ID: Final[str] = "flowing-mantis-239216"
# see start.sh
GCS_CRED_FILE: Final[str] = "~/.config/gcloud/application_default_credentials.json"
GCS_BUCKET: Final[str] = "gs://thesis-bucket-option-trade-classification/"

MODEL_DIR_LOCAL: Final[str] = "./models/"
MODEL_DIR_REMOTE: Final[str] = "models/"
