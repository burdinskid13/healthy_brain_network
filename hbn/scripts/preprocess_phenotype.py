import warnings
warnings.filterwarnings("ignore")


def run(parse=False):
    """Preprocess phenotypic data. Parses data from main file if it hasn't already been done.
    Creates new summary diagnosis file
    """
    import os
    import glob
    import pandas as pd
    from hbn.constants import Defaults
    from hbn.data import make_dataset
    from hbn.scripts import make_files

    print('preprocessing phenotype data', flush=True)
    if parse:
        # do some minimal preprocessing on the files (ONLY NEED TO DO THIS ONCE)
        assessments = ['Child_Measures', 'Parent_Measures', 'Clinical_Measures', 'Teacher_Measures']

        # loop over assessments
        for assessment in assessments:
            # save out assessments as separate csvs
            make_dataset.assessment_list(' '.join(assessment.split("_")), save=True)

            fdir = os.path.join(Defaults.PHENO_DIR, assessment)
            fpaths = glob.glob(f'{fdir}/*/*.csv')
            for fpath in fpaths:
                df = pd.read_csv(fpath)
                df['Identifiers'] = df['Identifiers'].str.strip(r',assessment|,,assessment|').str.extract(r'(\w+)', expand=False)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df = df[~df['Identifiers'].isna()]
                df.to_csv(fpath, index=False)

    # creates new clinical diagnosis file
    df = make_dataset.make_summary()
    print('created new clinical diagnosis file', flush=True)

    # make demographic features (saved in FEATURE_DIR)
    make_dataset.make_demographics()

    # make data files
    make_files.make_data_files()

    # makes test/train splits
    make_dataset.make_train_test_splits(out_dir=Defaults.MODEL_SPEC_DIR)

    return df


if __name__ == "__main__":
    run()


    