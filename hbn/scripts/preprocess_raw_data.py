import os
import warnings
warnings.filterwarnings("ignore")


def run(release='Release11_Apr2024'):
    """Preprocess phenotypic data
    All of the CSV files are manually downloaded from Loris and saved to `../hbn/phenotype/<release>`
    the following code sorts the files into their respective domain folders (e.g., Cognitive Testing) on the server
    """
    from hbn.constants import Defaults
    from hbn.scripts import make_train_test_split
    import glob
    import pandas as pd
    from hbn.data import make_dataset

    # parse phenotypic csv files
    make_dataset.parse_csv_files(
        release=release,
        data_dir=Defaults.PHENO_DIR)
    print(f'parsed phenotype data', flush=True)

    # make items file (maps data dict keys to column names of each csv file)
    make_dataset.make_items(
        data_dir=os.path.join(Defaults.PHENO_DIR, release), 
        outname='item-names.csv')
    print(f'create item-names.csv file', flush=True)

if __name__ == "__main__":
    run()


    