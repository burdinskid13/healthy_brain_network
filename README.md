Healthy Brain Networks
==============================

cerebellar fingerprints for neurodevelopmental disorders

Project Organization
------------

This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management

To install a virtualenv from the Pipfile:

    $ cd /om2/user/maedbh/healthy_brain_network
    $ pipenv install

To activate a virtualenv in order to access libraries:
    $ pipenv shell

### Data

> stored on OpenMind at `/nese/mit/group/sig/projects/hbn/phenotype`

    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.

### Code

> stored on OpenMind at `/om2/user/maedbh/healthy_brain_network`

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, model summaries and model spec files
    │
    ├── features           <- Feature spec files and csv files containing features (X) and target (y)
    │
    ├── hpc_scripts        <- Bash scripts for running jobs on openmind
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── Pipfile            <- The file for reproducing the analysis environment, e.g.
    │                         generated with `pipenv install` (install env) and `pipenv shell` (activate env)
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── hbn                <- Source code for use in this project.
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
    │   │   └── test_models.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │
    │   |
    │   └─── scripts <- Scripts to run workflow for phenotypic assessment
    │       └── run_phenotype_workflow.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
