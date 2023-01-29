![badge thesis](https://github.com/KarelZe/thesis/actions/workflows/action_latex.yaml/badge.svg)
![badge code](https://github.com/KarelZe/thesis/actions/workflows/action_python.yaml/badge.svg)
![badge code coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KarelZe/e2578f2f3e6322d299f1cb2e294d6b0b/raw/covbadge.json)

# thesis

## Overview

This repository contains all the resources for my thesis on option trade classification at Karlsruhe Institute of Technology.

| notes 📜                                                                                                                                                                   | schedule ⌚                                                                                                            | experiments 🧪                                                             | computing resources ☄️                                                                                                                                       | document 🎓                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| See [`references`](https://github.com/KarelZe/thesis/tree/main/references) folder. Download obsidian from [obsidian.md](https://obsidian.md/) to easily browse the notes. | Link to [tasks and mile stones](https://github.com/KarelZe/thesis/milestones?direction=asc&sort=due_date&state=open). | Link to [weights & biases](https://wandb.ai/fbv/thesis) (requires login). | Link to [gcp](https://console.cloud.google.com/welcome?project=flowing-mantis-239216) (requires login), and to [bwHPC](https://bwhpc.de/) (requires login). | see [`releases`](https://github.com/KarelZe/thesis/releases/). |

## Results ⚡
| year | gbm (classical) | gbm (classical + size) | gbm (classical + size + option) |
| ---- | --------------- | ---------------------- | ------------------------------- |
| 2015 | 60.39 (-2.98)   | 69.01 (5.64)           | 71.69 (8.32)                    |
| 2016 | 63.68 (-3.09)   | 72.52 (5.75)           | 74.45 (7.68)                    |
| 2017 | 64.56 (-3.63)   | 73.0 (4.81)            | 74.55 (6.36)                    |

(Comparison with best overall in [https://dx.doi.org/10.2139/ssrn.4098475](https://dx.doi.org/10.2139/ssrn.4098475). `(classical)` uses fewer features, `(classical + size)` identical features, `(classical + size + option)` more features.)

## How to use

Locally, or on [Jupyter cluster](https://uc2-jupyter.scc.kit.edu/jhub/):
```shell

# clone project
git clone https://github.com/KarelZe/thesis.git --depth=1

# set up consts for wandb + gcp
nano prod.env

# set up virtual env and install requirements
cd thesis

python -m venv thesis
source thesis/bin/activate
python -m pip install .

# run training script
python src/otc/models/train_model.py --trials=100 --seed=42 --model=gbm --dataset=fbv/thesis/ise_log_standardized:v2 --features=classical-size
2022-11-18 10:25:50,920 - __main__ - INFO - Connecting to weights & biases. Downloading artifacts. 📦
2022-11-18 10:25:56,180 - __main__ - INFO - Start loading artifacts locally. 🐢
2022-11-18 10:26:07,562 - __main__ - INFO - Start with study. 🦄
...
```

Using [`SLURM`](https://wiki.bwhpc.de/e/BwUniCluster2.0/Slurm) on [bwHPC](https://bwhpc.de/).
Set up `submit_thesis_gpu.sh` to batch job:
```shell
#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --partition=gpu_8 # See: https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues
#SBATCH --gres=gpu:1 # number of requested GPUs in node allocated by job
#SBATCH --time=10:00 # wall-clock time limit e. g. 10 minutes. Max is 48 hours.
#SBATCH --mem=128000 # memory in mbytes
#SBATCH --nodes=1 # no of nodes requested
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uxxxx@student.kit.edu

# Set up modules
module purge                    # Unload all currently loaded modules.
module load devel/python/3.8.6_gnu_10.2 # Load python 3.8 
module load devel/cuda/11.7     # Load required modules e. g., cuda 11.7

# start venv and run script
cd thesis

source thesis/bin/activate # Activate venv with dependencies already installed

python -u src/otc/models/train_model.py ...
```

Submit job:
```shell
# submit job to queue
sbatch ./submit_thesis_gpu.sh
Submitted batch job 21614924

# interact with job
scontrol show job

# view job logs
nano slurm-21614924.out
```

## Development

### Build and run docker image 🐳
The code is designed to run inside a docker container. See the [`Dockerfile`](https://github.com/KarelZe/thesis/blob/main/Dockerfile).
```shell
docker build -t thesis-dev .
docker run --env-file .env thesis-dev
```

### Set up git pre-commit hooks 🐙
Pre-commit hooks are pre-checks to avoid committing error-prone code. The tests are defined in the [`.pre-commit-config.yaml`](https://github.com/KarelZe/thesis/blob/main/.pre-commit-config.yaml). Install them using:
```shell
pip install .[dev]
pre-commit install
pre-commit run --all-files
```
### Run tests🧯
Tests can be run using [`tox`](https://tox.wiki/en/latest/). Just type:
```shell
tox
```
## Acknowledgement

The authors acknowledge support by the state of Baden-Württemberg through [bwHPC](https://bwhpc.de/).

Our implementation is based on:

<div class="csl-bib-body" style="line-height: 2; margin-left: 2em; text-indent:-2em;">
  <div class="csl-entry">Gorishniy, Y., Rubachev, I., Khrulkov, V., &amp; Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. <i>Advances in Neural Information Processing Systems</i>, <i>34</i>, 18932–18943.</div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Abook&amp;rft.genre=proceeding&amp;rft.atitle=Revisiting%20Deep%20Learning%20Models%20for%20Tabular%20Data&amp;rft.btitle=Advances%20in%20Neural%20Information%20Processing%20Systems&amp;rft.place=Red%20Hook%2C%20NY&amp;rft.publisher=Curran%20Associates%2C%20Inc.&amp;rft.aufirst=Yury&amp;rft.aulast=Gorishniy&amp;rft.au=Yury%20Gorishniy&amp;rft.au=Ivan%20Rubachev&amp;rft.au=Valentin%20Khrulkov&amp;rft.au=Artem%20Babenko&amp;rft.date=2021&amp;rft.pages=18932%E2%80%9318943&amp;rft.spage=18932&amp;rft.epage=18943"></span>
</div>
<div class="csl-bib-body" style="line-height: 2; margin-left: 2em; text-indent:-2em;">
  <div class="csl-entry">Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., &amp; Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. <i>Proceedings of the 32nd International Conference on Neural Information Processing Systems</i>, <i>32</i>, 6639–6649.</div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Abook&amp;rft.genre=proceeding&amp;rft.atitle=CatBoost%3A%20unbiased%20boosting%20with%20categorical%20features&amp;rft.btitle=Proceedings%20of%20the%2032nd%20International%20Conference%20on%20Neural%20Information%20Processing%20Systems&amp;rft.place=Red%20Hook%2C%20NY&amp;rft.publisher=Curran%20Associates%20Inc.&amp;rft.series=NeurIPS%202018&amp;rft.aufirst=Liudmila&amp;rft.aulast=Prokhorenkova&amp;rft.au=Liudmila%20Prokhorenkova&amp;rft.au=Gleb%20Gusev&amp;rft.au=Aleksandr%20Vorobev&amp;rft.au=Anna%20Veronika%20Dorogush&amp;rft.au=Andrey%20Gulin&amp;rft.date=2018&amp;rft.pages=6639%E2%80%936649&amp;rft.spage=6639&amp;rft.epage=6649"></span>
</div>
