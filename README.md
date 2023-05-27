![badge thesis](https://github.com/KarelZe/thesis/actions/workflows/action_latex.yaml/badge.svg)
![badge code](https://github.com/KarelZe/thesis/actions/workflows/action_python.yaml/badge.svg)
![badge code coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KarelZe/e2578f2f3e6322d299f1cb2e294d6b0b/raw/covbadge.json)

# thesis

## Overview

This repository contains all the resources for my thesis on option trade classification at Karlsruhe Institute of Technology.

| notes 📜                                                                                                                                                                   | schedule ⌚                                                                                                            | experiments 🧪                                                             | computing resources ☄️                                                                                                                                       | document 🎓                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| See [`references`](https://github.com/KarelZe/thesis/tree/main/references) folder. Download obsidian from [obsidian.md](https://obsidian.md/) to easily browse the notes. | Link to [tasks and mile stones](https://github.com/KarelZe/thesis/milestones?direction=asc&sort=due_date&state=open). | Link to [weights & biases](https://wandb.ai/fbv/thesis) (requires login). | Link to [gcp](https://console.cloud.google.com/welcome?project=flowing-mantis-239216) (requires login), and to [bwHPC](https://bwhpc.de/) (requires login). | see [`releases`](https://github.com/KarelZe/thesis/releases/). |

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
python src/otc/models/train_model.py --trials=100 --seed=42 --model=gbm --dataset=fbv/thesis/ise_supervised_log_standardized_clipped:latest --features=classical-size --pretrain
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
module load devel/cuda/10.2     # Load required modules e. g., cuda 10.2
module load compiler/gnu/11.2

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
<div class="csl-bib-body" style="line-height: 2; margin-left: 2em; text-indent:-2em;">
  <div class="csl-entry">Rubachev, I., Alekberov, A., Gorishniy, Y., &amp; Babenko, A. (2022). <i>Revisiting pretraining objectives for tabular deep learning</i> (arXiv:2207.03208). arXiv. <a href="http://arxiv.org/abs/2207.03208">http://arxiv.org/abs/2207.03208</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Adc&amp;rft.type=preprint&amp;rft.title=Revisiting%20pretraining%20objectives%20for%20tabular%20deep%20learning&amp;rft.description=Recent%20deep%20learning%20models%20for%20tabular%20data%20currently%20compete%20with%20the%20traditional%20ML%20models%20based%20on%20decision%20trees%20(GBDT).%20Unlike%20GBDT%2C%20deep%20models%20can%20additionally%20benefit%20from%20pretraining%2C%20which%20is%20a%20workhorse%20of%20DL%20for%20vision%20and%20NLP.%20For%20tabular%20problems%2C%20several%20pretraining%20methods%20were%20proposed%2C%20but%20it%20is%20not%20entirely%20clear%20if%20pretraining%20provides%20consistent%20noticeable%20improvements%20and%20what%20method%20should%20be%20used%2C%20since%20the%20methods%20are%20often%20not%20compared%20to%20each%20other%20or%20comparison%20is%20limited%20to%20the%20simplest%20MLP%20architectures.%20In%20this%20work%2C%20we%20aim%20to%20identify%20the%20best%20practices%20to%20pretrain%20tabular%20DL%20models%20that%20can%20be%20universally%20applied%20to%20different%20datasets%20and%20architectures.%20Among%20our%20findings%2C%20we%20show%20that%20using%20the%20object%20target%20labels%20during%20the%20pretraining%20stage%20is%20beneficial%20for%20the%20downstream%20performance%20and%20advocate%20several%20target-aware%20pretraining%20objectives.%20Overall%2C%20our%20experiments%20demonstrate%20that%20properly%20performed%20pretraining%20significantly%20increases%20the%20performance%20of%20tabular%20DL%20models%2C%20which%20often%20leads%20to%20their%20superiority%20over%20GBDTs.&amp;rft.identifier=http%3A%2F%2Farxiv.org%2Fabs%2F2207.03208&amp;rft.aufirst=Ivan&amp;rft.aulast=Rubachev&amp;rft.au=Ivan%20Rubachev&amp;rft.au=Artem%20Alekberov&amp;rft.au=Yury%20Gorishniy&amp;rft.au=Artem%20Babenko&amp;rft.date=2022-07-12"> </span>
</div>
