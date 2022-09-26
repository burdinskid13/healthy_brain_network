from pathlib import Path
import os
import numpy as np
import matplotlib.colors as mc
import colorsys

class Defaults:

    # set base directories
    BASE_DIR = Path(__file__).absolute().parent.parent # Path(__file__).absolute().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    INTERIM_DIR = DATA_DIR / "interim"
    PROCESSED_DIR = DATA_DIR / "processed"
    FIG_DIR = BASE_DIR / "reports" / "figures"
    PHENO_DIR = RAW_DIR / "phenotype"
    MODEL_DIR = INTERIM_DIR / "models"

    dirs = [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIG_DIR, MODEL_DIR]
    for dirn in dirs:
        if not os.path.isdir(dirn):
            os.makedirs(dirn)
            print(f'make new dir: {dirn}')
