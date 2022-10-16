![badge thesis](https://github.com/KarelZe/thesis/actions/workflows/action_latex.yaml/badge.svg)
![badge code](https://github.com/KarelZe/thesis/actions/workflows/action_python.yaml/badge.svg)

thesis
==============================

Code for thesis at Karlsruhe Institute of Technology

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── Graphs         <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── .pre-commit-config.yaml   <- Pre-checks to avoid committing error-prone code
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


Notes 📜
------------
Notes are stored in the `references` folder in an obsidian vault. Download obsidian from [obsidian.md](https://obsidian.md/) to easily browse the notes.
![obsidian](references/obsidian/img/obsidian.jpg)

mastermind board 🥷
------------
Link to [miro board](https://miro.com/app/board/uXjVPPRCa6s=/)

Final document 🎓
------------
The latest builds of the proposal and thesis can be found under [`releases`](https://github.com/KarelZe/thesis/releases/).

Set up pre-commit hooks 🐙
------------
```
pre-commit install
pre-commit run --all-files
```