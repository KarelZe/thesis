![badge thesis](https://github.com/KarelZe/thesis/actions/workflows/action_latex.yaml/badge.svg)
![badge code](https://github.com/KarelZe/thesis/actions/workflows/action_python.yaml/badge.svg)

# thesis

## Overview

This repository contains all the resources for my thesis on option trade classification at Karlsruhe Institute of Technology.

| notes 📜  |schedule ⌚   |mastermind board 🥷   |experiments 🧪   |computing resources ☄️   |document 🎓|
|---|---|---|---|---|---|
|See [`references`](https://github.com/KarelZe/thesis/tree/main/references) folder. Download obsidian from [obsidian.md](https://obsidian.md/) to easily browse the notes.   | Link to [tasks and mile stones](https://github.com/KarelZe/thesis/milestones?direction=asc&sort=due_date&state=open).  |Link to [miro board](https://miro.com/app/board/uXjVPPRCa6s=/) (requires login).   | Link to [weights & biases](https://wandb.ai/fbv/thesis) (requires login). |Link to [runpod](https://www.runpod.io/console/pods) (requires login) and to [gcp](https://console.cloud.google.com/welcome?project=flowing-mantis-239216) (requires login).|see [`releases`](https://github.com/KarelZe/thesis/releases/).|

## How to use

```shell

# clone project
git clone https://github.com/KarelZe/thesis.git --depth=1

# set up consts for wandb + gcp
nano /thesis/src/data/const.py

# install requirements
cd thesis
make requirements
pip install .

## run training script
python src/models/train_model.py --trials=25 --seed=42 --model=gbm --dataset=fbv/thesis/train_val_test_w_log_bin:v0 --features=ml

 dPYb,,dPYb,
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

2022-11-18 10:25:50,920 - __main__ - INFO - Connecting to weights & biases. Downloading artifacts. 📦
2022-11-18 10:25:56,180 - __main__ - INFO - Start loading artifacts locally. 🐢
2022-11-18 10:26:07,562 - __main__ - INFO - Start with study. 🦄
...
```

## Development

### Build and run docker image 🐳
The code is designed to run inside a docker container. See the [`Dockerfile`](https://github.com/KarelZe/thesis/blob/main/Dockerfile). 
```
docker build -t thesis-dev .
docker run --env-file .env thesis-dev
```

### Set up git pre-commit hooks 🐙
Pre-commit hooks are pre-checks to avoid committing error-prone code. The tests are defined in the [`.pre-commit-config.yaml`](https://github.com/KarelZe/thesis/blob/main/.pre-commit-config.yaml). Install them using:
```
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Acknowledgement
Our implementation is based on:

<div class="csl-bib-body" style="line-height: 2; margin-left: 2em; text-indent:-2em;">
  <div class="csl-entry">Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., &amp; Kasneci, G. (2022). <i>Deep Neural Networks and Tabular Data: A Survey</i> (arXiv:2110.01889). arXiv. <a href="http://arxiv.org/abs/2110.01889">http://arxiv.org/abs/2110.01889</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Adc&amp;rft.type=preprint&amp;rft.title=Deep%20Neural%20Networks%20and%20Tabular%20Data%3A%20A%20Survey&amp;rft.identifier=http%3A%2F%2Farxiv.org%2Fabs%2F2110.01889&amp;rft.aufirst=Vadim&amp;rft.aulast=Borisov&amp;rft.au=Vadim%20Borisov&amp;rft.au=Tobias%20Leemann&amp;rft.au=Kathrin%20Se%C3%9Fler&amp;rft.au=Johannes%20Haug&amp;rft.au=Martin%20Pawelczyk&amp;rft.au=Gjergji%20Kasneci&amp;rft.date=2022"></span>
</div>
<div class="csl-bib-body" style="line-height: 2; margin-left: 2em; text-indent:-2em;">
  <div class="csl-entry">Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., &amp; Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. <i>Proceedings of the 32nd International Conference on Neural Information Processing Systems</i>, <i>32</i>, 6639–6649.</div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Abook&amp;rft.genre=proceeding&amp;rft.atitle=CatBoost%3A%20unbiased%20boosting%20with%20categorical%20features&amp;rft.btitle=Proceedings%20of%20the%2032nd%20International%20Conference%20on%20Neural%20Information%20Processing%20Systems&amp;rft.place=Red%20Hook%2C%20NY&amp;rft.publisher=Curran%20Associates%20Inc.&amp;rft.series=NeurIPS%202018&amp;rft.aufirst=Liudmila&amp;rft.aulast=Prokhorenkova&amp;rft.au=Liudmila%20Prokhorenkova&amp;rft.au=Gleb%20Gusev&amp;rft.au=Aleksandr%20Vorobev&amp;rft.au=Anna%20Veronika%20Dorogush&amp;rft.au=Andrey%20Gulin&amp;rft.date=2018&amp;rft.pages=6639%E2%80%936649&amp;rft.spage=6639&amp;rft.epage=6649"></span>
</div>
