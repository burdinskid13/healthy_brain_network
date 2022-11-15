from pathlib import Path, PosixPath
import os

class Defaults:

    ## set directories for feature and model specs
    BASE_DIR = Path(__file__).absolute().parent.parent 
    FIG_DIR = BASE_DIR / "reports" / "figures"
    FEATURE_DIR = BASE_DIR / 'features'
    MODEL_SPEC_DIR = BASE_DIR / 'model_specs'
    BASH_SCRIPTS = BASE_DIR / 'hpc_scripts'
    TEST_DIR = BASE_DIR / "hbn" / 'tests'

    # set data base directories
    # DATA_DIR = PosixPath("/om2/user/shreyark/hbn_data") ## SET YOUR OWN PATH HERE
    DATA_DIR = BASE_DIR / 'data'
    RAW_DIR = DATA_DIR / "raw"
    INTERIM_DIR = DATA_DIR / "interim"
    PROCESSED_DIR = DATA_DIR / "processed"
    PHENO_DIR = RAW_DIR / "phenotype"
    MODEL_DIR = INTERIM_DIR / "models"

    dirs = [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIG_DIR, MODEL_DIR, FEATURE_DIR, MODEL_SPEC_DIR]
    for dirn in dirs:
        if not os.path.isdir(dirn):
            try:
                os.makedirs(dirn)
                print(f'make new dir: {dirn}')
            except:
                print(f'could not make {dirn}')
