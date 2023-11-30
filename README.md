Healthy Brain Network
==============================

Analysis of phenotypic data from the Child Mind Institute's Healthy Brain Network Initiative

First Steps
------------

* Clone Repo
> Clone the repo to your own path on OpenMind at **/om2/user/"username"**
```
git clone git@github.com:maedbhk/healthy_brain_network.git`
```

* Activate Virtual Environment
You can use either [**pipenv**](https://github.com/pypa/pipenv) or [**conda**](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for virtual environment and python package management
> see [OpenMind Setup](https://maedbhk.github.io/MIT-Projects/openmind/setup.html) for more detailed instructions on setting up virtual environments on OpenMind

* Install editable package (make sure virtual env is activated)
```
pip install -e .
```


* Activate jupyter notebook kernel
> To run jupyter notebook using modules installed in virtual env, run the following command in top-level directory of repo
```
ipython kernel install --name "hbn" --user
```

* Setting Paths and Accessing Data
> see [OpenMind Setup](https://maedbhk.github.io/MIT-Projects/mentorship/openmind.html) for more detailed instructions on setting paths on OpenMind
* Data are stored on OpenMind here: **/om2/user/maedbh/hbn_data**
* Create symlinks from this folder (or copy over **hbn_data/raw** folder) to your directory so you can read/write new files to your own path
Example Command:
```
cd /om2/user/"username"
mkdir hbn_data
cp -R /om2/user/maedbh/hbn_data/raw /om2/user/"username"/hbn_data/
```
* Go to **constants.py** and set __DATA_DIR__ to be the fullpath to your top-level directory of **hbn_data**
> For example: DATA_DIR = PosixPath("/om2/user/"username"/hbn_data")

* You can explore the HBN data dictionary `Release9_DataDic`, which is located on OpenMind at **hbn_data/raw/phenotype**

Example for creating specs:
------------
* feature specs are created using the following command:
    ```
    cd /om2/user/"username"/healthy_brain_network/hbn/scripts

    # preprocess phenotypic data
    python3 preprocess_phenotype.py

    # make features for modeling
    python3 make_phenotype_specs.py
    ```

Example for running modeling routine:
------------
* Model spec files (.json) are created using the following command:
    ```
    # make model specs
    python3 make_phenotype_models.py

    # run model pipeline
    python3 run_phenotype_models.py --cachedir=/om2/users/"username"/bin/.cache/pydra-ml/cache-wf/
    ``` 

* To run a predictive modeling script on OpenMind: 
    * 1) `cd  /om2/user/"username"/healthy_brain_network/hpc_scripts` 
    * 2) `vim test_phenotype_workflow.sh` and change the username
    * 3) Run the bash script: `sbatch test_phenotype_workflow.sh`
* The bash script executes the Python script **hbn/tests/test_workflow.py**

Project Organization
------------

### Data
> Note: **hbn_data** folder is stored on OpenMind at **/om2/user/maedbh/hbn_data/**

    ├── hbn_data
    │   ├── interim        <- Intermediate data that has been transformed (model outputs are stored here)
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump

### Directories
> PATHS are stored in **constants.py**: 

    ├── constants.py
    │   ├── DATA_DIR         <- top-level directory where **phenotype** folder is stored 
    │   ├── FEATURE_DIR      <- where feature spec (.json) and features (.csv) files are stored
    │   └── MODEL_SPEC_DIR   <- where model specs (.json) are stored
    │   └── MODEL_DIR        <- where model outputs (*pkl) are stored
    │   └── BASH_SCRIPTS     <- where bash scripts (.sh) are stored
    │   └── TEST_DIR         <- where test scripts (.py) are stored

### Code 

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
    │   │   └── run_phenotype_workflow.py
    │   │   └── feature_embeddings.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
