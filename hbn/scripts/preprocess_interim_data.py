
def get_all_participant_diagnoses(
    fpath, 
    outpath,
    demo_dir,
    dx_to_exclude=['PRem', 'RC', 'Rem', 'RuleOut']
    ):
    """ restructure `Diagnosis_Clinician_.csv` to get all participant diagnoses in one column (`all_dx`) - useful for generating train/test splits

    Args:
        fpath (str): fullpath to `Diagnosis_ClinicianConsensus.csv`
        outpath (str): fullpath to save out to disk. 
        demo_dir (str or None): full dir where demographic csv files are saved (e.g., `/raw/phenotype/<release>`)
    Returns: 
        dx_concat (pd dataframe)
    """
    import os
    from pathlib import Path
    from hbn.data import make_dataset
    import numpy as np
    import pandas as pd
    
    # read in clnical diagnosis file
    df = pd.read_csv(fpath, engine='python')

    # cleanup diagnosis cols
    df = make_dataset.cleanup_diagnosis_cols(df)

    # exclude diagnoses that have been ruled out, are in remission, or need confirmation
    df = make_dataset.exclude_diagnoses(df, dx_to_exclude)

    # add new categories (including categories to be modeled)
    df = make_dataset.define_new_categories(dataframe=df)

    # melt dataframe
    df = make_dataset.melt_dx(df, cols_to_keep=['Identifiers'])

    # add diagnosis groups that are useful later on
    df = make_dataset.add_diagnosis_groups(df)

    # add demographic info
    df_demos = make_dataset.get_demographics(
        df, 
        child_dir=os.path.join(demo_dir, 'child'),
        parent_dir=os.path.join(demo_dir, 'parent')
        )

    # save out data
    df_merged = df.merge(df_demos, on='Identifiers', how='outer')
    df_merged.to_csv(outpath, index=False)

    return df_merged


def remove_mixed_type_columns(dataframe):
    """ remove mixed type columns from dataframes - helps significantly with fitting models down the line...

    Args: 
        df (pd.DataFrame): dataframe to clean
    Returns: 
        df (pd.DataFrame): cleaned dataframe
    """
    import numpy as np

    def is_mixed_type_column(dataframe, col):
        """
        Returns a boolean indicating whether the specified column contains mixed data types (e.g., 'str', 'float') except for NaN.

        Args:
            df (pandas.DataFrame): The DataFrame containing the column to check.
            col (str): The column name to check for mixed data types.

        Returns:
            bool: True if the column contains mixed data types except for NaN, False otherwise.
        """
        if dataframe[col].dtype in [np.dtype("O")]:  # Check if the column is of string type
            for val in dataframe[col].dropna():  # Exclude NaN values
                if not isinstance(val, str) or val.startswith(("(", "{", "[")):  # Check if any value in the column is not of string type (except NaN) or starts with { ( [
                    return True
                else:
                    return False # If all values are of string type except NaN, the column is not mixed

    # remove mixed type columns
    cols = []
    for col in dataframe.columns:
        if is_mixed_type_column(dataframe, col):
            cols.append(col)
    df_out = dataframe.drop(cols, axis=1)

    return df_out, cols


def make_data_files(
    items_fpath,
    participant_fpath,
    data_dir,
    out_dir,
    participant_id='Identifiers',
    ):
    """ make data files and save in `Defaults.INTERIM_FEATURE_DIR`

    Args:   
        items_fpath (str): fullpath to `item-names-cleaned.csv`
        participant_fpath (str): fullpath to `all_participant_diagnoses.csv`
        data_dir (str): fullpath to where csv files are saved
        out_dir (str): directory where output files will be saved
        participant_id (str): column name of participant identifiers. default is 'Identifiers'

    Returns:
        saves out interim data files to `Defaults.INTERIM_FEATURE_DIR`
    """
    import os
    import pandas as pd
    import logging

    # make data files
    df_all = pd.DataFrame()
    for assessment in ['child', 'parent', 'teacher']:

        # load participant info
        df_participants = pd.read_csv(participant_fpath, engine='python')

        # load items info
        df_items = pd.read_csv(items_fpath, engine='python')

        # grab all assessment csv files corresponding to `assessment`
        df_items_assessment = df_items[df_items['assessment']==assessment]

        # loop over 
        questionnaires = df_items_assessment['abbrev'].unique()
        identifiers = df_participants[participant_id].unique()

        filters = ['raw', 'Scores', 'Question']
        for filter_col in filters:
            df_csv_all = pd.DataFrame(identifiers, columns=[participant_id])
            for quest in questionnaires:
                df_quest = df_items_assessment[df_items_assessment['abbrev']==quest]
                if 'raw' not in filter_col:
                    df_quest = df_quest[df_quest['col_type']==filter_col]

                # load csv file
                if not df_quest.empty:
                    domain = df_quest['domain'].unique()[0]
                    abbrev = df_quest['abbrev'].unique()[0]
                    df_csv = pd.read_csv(os.path.join(data_dir, assessment, domain, f'{quest}.csv'), engine='python')

                    # index the csv files with the data dictionary files
                    cols_to_keep = df_quest[df_quest['data_col_name'].notna()]['data_col_name'].tolist()
                    cols_to_keep_full = [participant_id] + [f'{abbrev},{col}' for col in cols_to_keep]
                    df_csv = df_csv[cols_to_keep_full]

                    if len(cols_to_keep_full)==1:
                        logging.basicConfig(filename=os.path.join(out_dir, 'questionnaires-not-logged.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                        logging.info(f'{quest} not added to dataframe for {filter_col}')

                    # merge files
                    df_csv_all = df_csv_all.merge(df_csv, on='Identifiers', how='left')
                else:
                    logging.basicConfig(filename=os.path.join(out_dir, 'questionnaires-not-logged.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                    logging.info(f'{quest} is empty for {filter_col} items')

            # save out  
            df_csv_all.to_csv(os.path.join(out_dir, f'{assessment}-features-{filter_col}.csv'), index=False)


def run(release='Release11_Apr2024'):
    import os
    from hbn.constants import Defaults
    from hbn.scripts import make_train_test_split

    release='Release11_Apr2024'
    clinical_dir = os.path.join(Defaults.PHENO_DIR, release, 'clinician', 'Diagnosis')
    interim_dir = os.path.join(Defaults.INTERIM_FEATURES_DIR, release)
    if not os.path.isdir(interim_dir):
        os.makedirs(interim_dir)

    # get all participant diagnoses
    preprocess_interim_data.get_all_participant_diagnoses(
        fpath=os.path.join(clinical_dir, 'Diagnosis_ClinicianConsensus.csv'),
        outpath=os.path.join(interim_dir, 'all_participant_diagnoses.csv'),
        demo_dir=os.path.join(Defaults.PHENO_DIR, release)
        )
    print('created participant diagnosis file', flush=True)

    # make parent files
    preprocess_interim_data.make_data_files(
        items_fpath=os.path.join(Defaults.PHENO_DIR, release, 'item-names.csv'),
        participant_fpath=os.path.join(interim_dir, 'all_participant_diagnoses.csv'),
        data_dir=os.path.join(Defaults.PHENO_DIR, release),
        out_dir=interim_dir
        )
    print('created new data files', flush=True)

    # makes test/train splits
    make_train_test_split.run(inpath=os.path.join(interim_dir, 'all_participant_diagnoses.csv'), 
                              outpath=os.path.join(interim_dir, 'participant_train_test.csv')
                              )
    print('created train/test participant files', flush=True)

if __name__ == "__main__":
    run()
    

    
