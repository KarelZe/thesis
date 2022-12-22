As we aim for reproducability and configuration. We give a detailed description about the environment in which the experiments were conducted.

> We recommend that authors state the name of the software and specific release version they used, and also cite the paper describing the tool. In cases when an updated version of a tool used has been published, it should be cited to ensure that the precise version used is clear. And when there is doubt, an e-mail to the developers could resolve the issue.
(found in [[@GivingSoftwareIts2019]])

**Data:** Data set versioning using `SHA256` hash. Data is loaded by key using `wandb` library.
**Hardware:** Training and inference of our models is performed on the bwHPC cluster. Each node features ... cpus, ... gpus with cuda version ... and Ubuntu ... . If runpods (https://www.runpod.io/gpu-instance/pricing) or lambdalabs (https://lambdalabs.com/service/gpu-cloud) is used, cite it as well. If I use my docker image cite as well. Model training of gradient boosting approach and transformers is performed on accelerates. To guarantee deterministic behaviour (note gradient boosting may operate on samples, initializations of weights happens randomly, cuda runtime may perform non-deterministic optimizations) we fix the random seeds.
**Software:** Python version (...). Cite packages as "pandas (v1.5.1)". List software packages:
```python
# copied from repo at 12/08/2022
dependencies = [
  "click==8.1.3",
  "catboost==1.1",
  "einops==0.6.0",
  "fastparquet==2022.12.0",
  "gcsfs==2022.11.0",
  "google-auth==2.15.0", # needed by w&b in runpod
  "modin==0.17.0",
  "numpy==1.23.4",
  "optuna==3.0.3",
  "pandas==1.5.1",
  "pandas-datareader==0.10.0",
  "psutil==5.9.4", #needed by w&b
  "pydantic==1.10.2",
  "python-dotenv>=0.5.1",
  "pyyaml==6.0",
  "requests==2.28.1",
  "scikit-learn==1.1.3",
  "seaborn==0.12.1",
  "shap==0.41.0",
  "torch==1.13.0",
  "tqdm==4.64.1",
  "typer==0.7.0",
  "wandb==0.13.5",
]
```
- For presentation of chapter see: [[@prokhorenkovaCatBoostUnbiasedBoosting2018]]
- source code of experiments and paper is available at https://github.com/KarelZe/thesis/
- Get some inspiration from https://madewithml.com/#mlops