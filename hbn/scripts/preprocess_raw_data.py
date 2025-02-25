import os
import warnings
warnings.filterwarnings("ignore")

def parse_raw_data(assessment, release, data_dir):
    import glob
    import pandas as pd
    from hbn.data import make_dataset

    """All of the CSV files are manually downloaded from Loris and are sorted into their respective domain folders (e.g., Cognitive Testing) on the server
    """

    # save out assessments as separate csvs
    make_dataset.separate_main_assessment_file_into_csvs(
        assessment, 
        fpath=os.path.join(data_dir, 'Assessment_List_Jan2019.xlsx'), 
        data_dir=data_dir)

    # make `Free_Assessments_HBN_new.csv` file
    make_dataset.make_new_proprietary_assessments_file(
        filename='Free_Assessments_HBN.xlsx', 
        outname='Free_Assessments_HBN_new.csv',
        data_dir=os.path.join(data_dir, release), 
        )
    print('created proprietary assessments file', flush=True)

    # make items file
    make_dataset.make_items_file(
        filename='item-names.csv', 
        proprietary_filename='Free_Assessments_HBN_new.csv',
        outname='item-names-cleaned.csv',
        data_dir=os.path.join(data_dir, release),
    )
    print('created new item names file', flush=True)


def run(release='Release9_Nov2020'):
    """Preprocess phenotypic data. Parses data from main file if it hasn't already been done.
    Creates new summary diagnosis file
    """
    from hbn.constants import Defaults
    from hbn.scripts import make_train_test_split

    # assessments to parse
    assessments = ['Child_Measures', 'Parent_Measures', 'Clinical_Measures', 'Teacher_Measures']

    # loop over assessments and parse
    for assessment in assessments:
        parse_raw_data(assessment, release, data_dir=Defaults.PHENO_DIR)
        print(f'parsed phenotype data for {assessment}', flush=True)

    # makes test/train splits
    make_train_test_split.run(inpath=os.path.join(Defaults.INTERIM_FEATURES_DIR, release, 'all_participant_diagnoses.csv'), 
                              outpath=os.path.join(Defaults.INTERIM_FEATURES_DIR, release, 'participant_train_test.csv')
                              )
    print('created train/test participant files', flush=True)


if __name__ == "__main__":
    run()


    