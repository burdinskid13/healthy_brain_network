import click
import warnings
warnings.filterwarnings("ignore")

def run(
    transformer='all-MiniLM-L6-v2'
    ):
    import os
    import pandas as pd
    import numpy as np
    from hbn.constants import Defaults
    from hbn.models import item_analysis
    from hbn.scripts import make_files
    from hbn import io

    # load data
    print("loading data dictionary", flush=True)
    fpath = os.path.join(Defaults.PHENO_DIR, 'item-names-cleaned.csv')
    dataframe = pd.read_csv(fpath)

    # remove NaN values from dataframe
    idx = dataframe['questions'].isna()
    df = dataframe[~idx].reset_index(drop=True)

    # calculate similarity between clinical questionnaires
    print(f'calculating item analysis on {transformer}...', flush=True)
    list_of_dicts = item_analysis.sentence_similarity(data_dictionary=df, transformer=transformer)

    # save as hdf5
    fname = 'sentence-similarity_HBN_' + transformer + '.h5'
    io.save_dict_as_hdf5(fpath=os.path.join(Defaults.SUBTYPE_DIR, fname), data_dict=list_of_dicts)

if __name__ == "__main__":
    run()
