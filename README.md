Healthy Brain Networks
==============================

cerebellar fingerprints for neurodevelopmental disorders

First Steps
------------
This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management
> see [OpenMind Setup](https://maedbhk.github.io/MIT-Projects/openmind/setup.html) for more detailed instructions on setting up virtual environments on OpenMind

To run jupyter notebook using modules installed in virtual env, run the following command in top-level directory of repo
> `ipython kernel install --name "hbn" --user`

Data Exploration
------------
* To explore and visualize clinical diagnoses, check out **notebooks/clinical_dx.ipynb**
    * if you're having difficulty opening notebooks on OpenMind, then you can always explore the data yourself by loading dataframe and using seaborn or plotly to do some visualizations
    ```
    from hbn.data import make_dataset
    
    df, _ = make_dataset.get_clinical_diagnosis(demographics=True, target=None)
    ```
* To visualize output of predictive modeling, check out **notebooks/phenotype_models.ipynb.ipynb**

Features
------------
* features (including X variables and y target variable) are created from a **spec file** using the following command:
    ```
    from hbn.features import build_features 
    
    build_features.make_features(spec_file)
    ```
    * for an example of a feature spec file (.json) and features file (.csv), see example files in **hbn/tests/data** with a more detailed description in **README**

Predictive Modeling
------------
* Model spec files (.json) are created using the function **hbn.models.first_level_modeling.make_specs**
    * for an example of a model spec file (.json), see example file in **hbn/tests/data** with a more detailed description in **README**
* To run a predictive modeling script on OpenMind: 
    * 1) `cd hpc_scripts` 2) `vim run_phenotype_workflow_openmind.sh` and change the **cachedir** input to point to your cache directory,  and run the bash script: 3) `sbatch run_phenotype_workflow_openmind.sh`
    * The bash script executes the Python script **test_workflow.py**

Project Organization
------------

### Data

    ├── phenotype
    │   ├── interim        <- Intermediate data that has been transformed (model outputs are stored here)
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump

* **data** are stored on OpenMind at **/nese/mit/group/sig/projects/hbn/phenotype**

### Code
> Clone the repo to your own path on OpenMind at **/om2/user/"username"/** (example: **/om2/user/"username"/healthy_brain_network**)

* PATHS are stored in **constants.py**: 
    * **DATA_DIR**: top-level directory where **phenotype** data folders are stored 
    * **FEATURE_DIR**: where feature spec files (.json) and csv files are stored
    * **MODEL_SPEC_DIR**: where model specs (.json) are stored
    * **hpc_scripts**: where bash scripts are stored (running on OpenMind)

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── model_specs        <- Model Spec files
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
    |   |
    |   ├── constants.py      <- Directories are set here
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
