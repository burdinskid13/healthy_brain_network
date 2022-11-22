import warnings
warnings.filterwarnings("ignore")


def run():
    """Preprocess phenotypic data. Parses data from main file if it hasn't already been done.
    Creates new summary diagnosis file
    """
    import os
    from hbn.constants import Defaults
    from hbn.data import make_dataset

    assessments = ['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures']
    parent_file = os.path.join(Defaults.PHENO_DIR, 'data-2022-08-24T16_37_18.263Z.csv')

    # parses data if hasn't already been done
    for assessment in assessments:
        fdir = os.path.join(Defaults.PHENO_DIR, '_'.join(assessment.split()))
        if not os.path.isdir(fdir):
            make_dataset.parse_phenotypic_data(
                parent_file=parent_file,
                assessment=assessment, 
                out_dir=Defaults.PHENO_DIR
                )
        print('phenotypic data have already been parsed...')

    # create new questionnaires from Parents Intake Interview
    make_dataset.parse_intake_interview()

    # creates new clinical diagnosis file
    df = make_dataset.make_summary()
    print('created new clinical diagnosis file')

    # makes test/train splits
    make_dataset.make_train_test_splits(out_dir=Defaults.MODEL_SPEC_DIR)

    return df


if __name__ == "__main__":
    run()


    