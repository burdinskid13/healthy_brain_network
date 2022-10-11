Healthy Brain Networks
==============================

cerebellar fingerprints for neurodevelopmental disorders

First Steps
------------
This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management
> see [OpenMind Setup](https://maedbhk.github.io/MIT-Projects/openmind/setup.html) for more detailed instructions on setting up virtual environments on OpenMind

Predictive Modeling
------------
* The following command runs the Python predictive modeling script: `hbn/scripts/run_phenotype_workflow.py`
    * See `hpc_scripts/run_phenotype_workflow_openmind.sh` for running an example slurm script on OpenMind

Project Organization
------------

### Data
> stored on OpenMind at `/nese/mit/group/sig/projects/hbn/phenotype`

    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.

### Code
> Clone the repo to your own path on OpenMind at `/om2/user/<username>/` (example: `/om2/user/maedbh/healthy_brain_network`)

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Model Spec files
    │
    ├── features           <- Feature spec files and csv files containing features (X) and target (y)
    │
    ├── hpc_scripts        <- Bash scripts for running jobs on openmind. See `run_phenotype_workflow_openmind.sh` as an example
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
    │                         generated with `$ pipenv install` (to install environment) and `$ pipenv shell` (to activate environment)
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
    │   │   └── second_level_modeling.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   |
    │   └─── scripts <- Scripts to run workflow for phenotypic assessment
    │       └── run_phenotype_workflow.py
    │   │   └── feature_embeddings.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
