Healthy Brain Networks
==============================

cerebellar fingerprints for neurodevelopmental disorders

First Steps
------------
This project uses [**pipenv**](https://github.com/pypa/pipenv) for virtual environment and python package management
> see [OpenMind Setup](https://maedbhk.github.io/MIT-Projects/openmind/setup.html) for more detailed instructions on setting up virtual environments on OpenMind

Clone the repo to your own path on OpenMind at **/om2/user/"username"**
> `git clone git@github.com:maedbhk/healthy_brain_network.git`

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
* To visualize output of predictive modeling, check out **notebooks/phenotype_models.ipynb**

Features
------------
* features (including X variables and y target variable) are created from a **spec file** using the following command:
    ```
    import os
    from hbn.features import build_features 

    # example feature_spec
    feature_spec = os.path.join(Defaults.TEST_DIR, 'features-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize-spec.json')
    build_features.make_features(feature_spec, out_dir=Defaults.FEATURE_DIR)
    ```
* for an example of a feature spec file (.json) and features file (.csv), see example files in **hbn/tests/data** with a more detailed description in **hbn/tests/README**

Predictive Modeling
------------
* Model spec files (.json) are created using the following command:
    ```
    from hbn.models import first_level_modeling
    
    # example feature_spec
    feature_spec = os.path.join(Defaults.TEST_DIR, 'features-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize-spec.json')
    first_level_modeling.make_specs(feature_spec, out_dir=Defaults.MODEL_SPEC_DIR)
    
    ``` 
* For an example of a model spec file (.json), see example file in **hbn/tests/data** with a more detailed description in **README**
* To run a predictive modeling script on OpenMind: 
    * 1) `cd hpc_scripts` 
    * 2) `vim run_phenotype_workflow_openmind.sh` and change the **cachedir** input to point to your cache directory 
    * 3) Run the bash script: `sbatch run_phenotype_workflow_openmind.sh`
* The bash script executes the Python script **test_workflow.py**

Project Organization
------------

### Data

    ├── phenotype
    │   ├── interim        <- Intermediate data that has been transformed (model outputs are stored here)
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump

* **phenotype** folder is stored on OpenMind at **/nese/mit/group/sig/projects/hbn/**

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
