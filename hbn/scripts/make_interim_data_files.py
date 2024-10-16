import os
import pandas as pd
import glob
from pathlib import Path

from hbn.data.make_dataset import *
from hbn.features.build_features import *
from hbn.constants import Defaults

def separate_main_assessment_file_into_csvs(
        assessments=['Child_Measures', 'Parent_Measures', 'Clinical_Measures', 'Teacher_Measures'], 
        fpath=os.path.join(Defaults.PHENO_DIR, 'Assessment_List_Jan2019.xlsx'), 
        data_dir=Defaults.PHENO_DIR
        ):
    """correct assessment list, update `domain` for each `measure`

    Args:
        assessment (list of str): default is: ['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures']
        assessment_file (str): full path to `Assessment_List_Jan2019.xlsx`
        data_dir (str): directory where output will be saved. Default is `PHENO_DIR`
    Returns: 
        info (pd dataframe), domain (str)
    """
    # parse `assessment` sheets from `fpath`
    xls = pd.ExcelFile(fpath, engine='openpyxl');

    # loop over individual assessments
    for assessment in assessments:

        # save out assessments as separate csvs
        assessment_split = ' '.join(assessment.split("_"))
        
        # read excel
        info = pd.read_excel(xls, assessment_split, header=1);
        info = info.dropna(axis=0, how='all') # drop rows that are all NaN
        
        # populate NaN entries with correct Domain
        domain = False
        if 'Domain' in info.columns:
            domain = True; domains = []
            for row in info.index:
                row_value = info.loc[row, 'Domain']
                if type(row_value)==str:
                    domains.append(row_value)
                else:
                    info.loc[row, 'Domain'] = domains[-1]
        
        # save out individual assessment as csv
        fname = Path(fpath).stem # remove .xlsx
        assessment_out = '_'.join(assessment.split())
        info.to_csv(os.path.join(data_dir, f'{fname}_{assessment_out}.csv'))


def remove_redundant_identifiers_from_assessments(
        assessments=['Child_Measures', 'Parent_Measures', 'Clinical_Measures', 'Teacher_Measures'], 
        data_dir=Defaults.PHENO_DIR):
    """ some basic clean up on assessment files, remove redundant identifiers from assessment csvs

    Args:
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'
        data_dir (str): directory where output will be saved. Default is `PHENO_DIR`
    """
    # loop over assessments
    for assessment in assessments:
        # load assessment
        fdir = os.path.join(data_dir, assessment)
        # get all csv files within assessment directory
        fpaths = glob.glob(f'{fdir}/*/*.csv')
        # loop over files
        for fpath in fpaths:
            df = pd.read_csv(fpath, engine='python')
            df['Identifiers'] = df['Identifiers'].str.strip(r',assessment|,,assessment|').str.extract(r'(\w+)', expand=False)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df[~df['Identifiers'].isna()]
            df.to_csv(fpath, index=False)


def make_clinical_summary_file(
    filename='Diagnosis_ClinicianConsensus.csv', 
    outname='Clinical_Diagnosis_Demographics.csv',
    data_dir=os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures')
    ):
    """
    Preprocess `Diagnosis_ClinicianConsensus.csv` and save out as `Clinical_Diagnosis_Demographics.csv`. Also, save out participant identifiers to `data_dir`
    `data_dir` should contain `filename`. `outname` will also be saved to `data_dir`
    Args:
        filename (str): fullpath to `Diagnosis_ClinicianConsensus.csv`
        outname (str or None): filename to save out to disk
        data_dir (str or None): directory where will `Clinical_Diagnosis_Demographics` and `participants` be saved. If None, no files are saved out.
    Returns: 
        dx (pd dataframe)
    """
    from pathlib import Path
    
    # READ CLINICAL CONSENSUS
    dx = pd.read_csv(os.path.join(data_dir, filename), engine='python')

    # do some clean up
    dx.columns = dx.columns.str.replace('Diagnosis_ClinicianConsensus,', '')

    # replace NaN
    for num in range(1,11):
        # num
        num_str = str(num).zfill(2)
        
        # Replace "NaN" diagnosis with 'No Diagnosis Given: No Reason Given'
        dx.loc[dx[f'DX_{num_str}']==' ',f'DX_{num_str}'] = 'No Diagnosis Given: No Reason Given'
        dx.loc[dx[f'DX_{num_str}_Cat'].isna(),f'DX_{num_str}_Cat'] = 'No Diagnosis Given: No Reason Given'

    # new disorder category
    diagnoses = [f'DX_{f:02}' for f in np.arange(1,11)]
    dx['comorbidities'] = dx[diagnoses].count(axis=1)-1

    # add demographics
    dx = add_demographics(dataframe=dx)

    # bucket ages: early, emerging, and fluent readers
    dx.loc[dx['Age']>=10, 'Age_bracket'] = "over10"
    dx.loc[dx['Age']<10, 'Age_bracket'] = "under10"

    dx['Age_round'] = dx['Age'].round()

    # deal with missing values and NaN
    dx = dx.replace(' ', np.float("NaN")).fillna(np.float("NaN")).dropna(how='all', axis=1)
    dx = dx.dropna(how='all', axis=0)

    # add new categories (including categories to be modeled)
    dx = define_new_categories(dataframe=dx)

    # add ethnicity
    dx = add_race_ethnicity(dataframe=dx)

    # participants
    dx = dx.loc[:, ~dx.columns.str.contains('^Unnamed')]
    
    # save out new files to disk
    if data_dir is not None:
        dx['Identifiers'].to_csv(os.path.join(data_dir, 'participants.csv'))
        outpath = os.path.join(data_dir, outname)
        dx.to_csv(outpath, index=False)

    return dx


def get_all_participant_diagnoses(
    fpath, 
    outname='all_participant_diagnoses.csv',
    data_dir=Defaults.INTERIM_FEATURES_DIR,
    dx_to_exclude=['PRem', 'RC', 'Rem', 'RuleOut']
    ):
    """ restructure `Clinical_Diagnosis_Demographics.csv` to get all participant diagnoses in one column (`all_dx`) - useful for generating train/test splits

    Args:
        fpath (str): fullpath to `Clinical_Diagnosis_Demographics.csv`
        outname (str or None): filename to save out to disk. default is `all_participant_diagnoses.csv`
        data_dir (str or None): directory where data are stored and `all_participant_diagnoses.csv` will be saved. If None, no files are saved out.
    Returns: 
        dx_concat (pd dataframe)
    """
    # read in clnical diagnosis file
    df = pd.read_csv(fpath, engine='python')

    def exclude_diagnoses(df, dx_to_exclude):
        # we're excluding diagnoses that have the following labels: RuleOut, Rem (remission), PRem (partial remission), RC (requires confirmation)
        # see full list here: https://docs.google.com/spreadsheets/d/1si0JDiI0rELnyQAGAaoO3lbERRfcKQ5X/edit?usp=sharing&ouid=115304373382106482578&rtpof=true&sd=true
        # we're doing this by setting these diagnoses to NaN
        for dx in dx_to_exclude:
            for num in np.arange(1,11):
                for col in [f'DX_{num:02d}', f'DX_{num:02d}_Cat', f'DX_{num:02d}_Cat_new']:
                    df.loc[df[f'DX_{num:02d}_{dx}']==1, col] = np.nan
        return df

    # exclude diagnoses that have been ruled out, are in remission, or need confirmation
    df = exclude_diagnoses(df, dx_to_exclude)

    cols_to_keep = ['Identifiers', 'PreInt_Demos_Fam,Child_Race_cat', 'Sex', 'Age_round']

    dx_cols = [f'DX_{num:02d}' for num in np.arange(1,11)]
    dx_subtype = pd.melt(df, id_vars=cols_to_keep, value_vars=dx_cols, var_name='DX_Subtype', value_name='DX_Subtype_Name')

    dx_cols = [f'DX_{num:02d}_Cat_new' for num in np.arange(1,11)]
    dx_cat = pd.melt(df, id_vars=cols_to_keep, value_vars=dx_cols, var_name='DX_Cat', value_name='DX_Cat_Name')

    dx_concat = pd.concat([dx_cat, dx_subtype[['DX_Subtype', 'DX_Subtype_Name']]], axis=1)
    
    # remove duplicate columns
    dx_concat = dx_concat.loc[:, ~dx_concat.columns.duplicated()].copy()

    # save out data
    if data_dir is not None:
        outpath = os.path.join(data_dir, outname)
        dx_concat.to_csv(outpath, index=False)

    return dx_concat


def make_new_proprietary_assessments_file(
    data_dir=Defaults.PHENO_DIR,
    filename='Free_Assessments_HBN.xlsx', 
    outname='Free_Assessments_HBN_new.csv'
    ):
    """
    Preprocess `Free_Assessments_HBN.xlsx` and save out as `Free_Assessments_HBN_new.csv`.
    `data_dir` should contain `filename`. `outname` will also be saved to `data_dir`
    Args:
        data_dir (str): directory where `filename` is saved and where `outname` will be saved
        filename (str or None): `Free_Assessments_HBN.xlsx` file. 
        outname (str or None): saves out as `Free_Assessments_HBN_new.csv`. 
    """

    # get full path to proprietary data
    df = pd.read_excel(os.path.join(data_dir, filename))

    # get outpath
    outpath = os.path.join(data_dir, outname)

    # rows to be added to the dataframe
    add_rows = [
        {'Assessment': 'Adverse Childhood Experiences Scale (ACE_P)', 'Price': 'Free', 'used_in_study': 'HBN'},
        {'Assessment': 'Alabama Parenting Questionnaire – Self Report (APQ_SR)', 'Price': 'Free', 'used_in_study': 'HBN'},
        {'Assessment': 'Barratt Simplified Measure of Social Status (Barratt)', 'Price': 'Proprietary', 'used_in_study': 'HBN, NKI Rockland'},
        {'Assessment': 'Conners 3 - Self-Report (C3SR)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Child Behavior Checklist - Pre-School (CBCL_Pre)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Teacher Report Form Preschool Age (TRF_P)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Teacher Report Form School Age (TRF)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Clinical Evaluation of Language Fundamentals, Fifth Edition Screener (CELF)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Clinical Evaluation of Language Fundamentals, Fifth Edition Full Assessment (CELF_Full_5to8)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Clinical Evaluation of Language Fundamentals, Fifth Edition Full Assessment (CELF_Full_9to21)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Clinical Evaluation of Language Fundamentals, Fifth Edition Metalinguistics (CELF_Meta)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Child Flourishing (CFS)', 'Price': 'Unknown', 'used_in_study': 'HBN'},
        {'Assessment': 'Ishihara Color Vision Test (ColorVision)', 'Price': 'Unknown', 'used_in_study': 'HBN'},
        {'Assessment': 'Comprehensive Test of Phonological Processing (CTOPP)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Dishion Social Acceptance Scale - Teacher (Dishion_Teacher)', 'Price': 'Unknown', 'used_in_study': 'HBN'},
        {'Assessment': 'Expressive Vocabulary Test (EVT)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Internet Use Questionnaire Parent (IUQ_P)', 'Price': 'Free', 'used_in_study': 'HBN'},
        {'Assessment': 'Internet Use Questionnaire Self-Report (IUQ_SR)', 'Price': 'Free', 'used_in_study': 'HBN'},
        {'Assessment': 'Kaufman Brief Intelligence Test (KBIT)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'National Institute of Health Toolbox Full Data (NIH_Full)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'National Institute of Health Toolbox Full Data (NIH_Scores)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Negative Life Events Scale Self Report (NLES_SR)', 'Price': 'Free', 'used_in_study': 'HBN'},
        {'Assessment': 'The Positive and Negative Affect Schedule (PANAS)', 'Price': 'Free', 'used_in_study': 'HBN'},
        {'Assessment': 'Positive Behavior Scale (PBS)', 'Price': 'Unknown', 'used_in_study': 'HBN'},
        {'Assessment': 'Screen for Anxiety Related Disorders Self Report (SCARED_SR)', 'Price': 'Free', 'used_in_study': 'HBN'},
        {'Assessment': 'TOWRE-2 (TOWRE)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Vineland Adaptive Behavior Scale-II (Vineland)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Wechsler Adult Intelligence Scale (WAIS)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Wechsler Adult Intelligence Scale (WAIS_abb)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Wechsler Abbreviated Scale of Intelligence (WASI)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Wechsler Individual Achievement Test (WIAT)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Wechsler Intelligence Scale for Children (WISC)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Grooved Pegboard (Pegboard)', 'Price': 'Proprietary', 'used_in_study': 'HBN'},
        {'Assessment': 'Yale Food Addiction Scale (YFAS_C)', 'Price': 'Free', 'used_in_study': 'HBN'},
        ]

    remap = {
        'APQ _ Parent': 'APQ_P',
        'APQ – Parent': 'APQ_P',
        'CBCL _ TRF': 'TRF',
        'C_SSRS': 'CSSRS',
        'NLES _ Parent': 'NLES_P',
        'E_SWAN': 'ESWAN',
        'PSITM': 'PSI',
        'RBS_R': 'RBS',
        'SCARED': 'SCARED_P',
        'SDSC': 'SDS',
        'SRS_P': 'SRS_Pre',
        'SRS_2': 'SRS',
        'Symptom Checker': 'SympChck',
        }

    # need to match the keys in `Free_Assessments_HBN.xlsx` to the dataframe
    def remap_keys(x):
        if x in list(remap.keys()):
            return remap[x]
        else:
            return x

    # add new cols to dataframe
    for row in add_rows:
        df.loc[len(df.index)] = list(row.values())

    # make new cols `measure` and `datadic`
    df['measure'] = df['Assessment'].str.split('(').str.get(0)
    df['datadic'] = df['Assessment'].str.split('(').str.get(1).str.replace(')', '')

    # replace '-' with '_' in datadic
    df['datadic'] = df['datadic'].str.replace('-', '_')

    # match the keys in `Free_Asssessments_HBN.xlsx` to the dataframe
    df['datadic'] = df['datadic'].apply(lambda x: remap_keys(x))

    # Convert NaN values to Unknown
    df.loc[df['Price'].isna(), 'Price'] = 'Unknown'

    # create boolean columns for proprietary/free questionnaires
    df['Free_Assessments'] = np.where(df['Price'] =='Free', True, False)
    df['Proprietary_Assessments'] = np.where(df['Price'] =='Proprietary', True, False)

    # save to file
    df.to_csv(outpath, index=False)

    return df


def make_items_file(
    filename='item-names.csv', 
    proprietary_filename='Free_Assessments_HBN_new.csv',
    outname='item-names-cleaned.csv',
    data_dir=Defaults.PHENO_DIR,
    ):
    """preprocess `item-names.csv` and save out as `item-names-cleaned.csv` to `out_dir` (modified from original https://github.com/charlie42/diagnosis-predictor/blob/main/references/item-names.csv)

    `filename` and `proprietary_filename` should exist in `data_dir`

    Args:
        filename (str): filename `item-names.csv`
        proprietary_filename (str or None): optionally merge columns from `Free_Assessments_HBN_new.csv` to `filename`. 
        outname (str): filename `item-names-cleaned.csv` where preprocessed items will be saved.
        data_dir (str): directory where `item-names.csv` `Free_Assessments_HBN_new.csv` are located
    """
    # load item names filename
    dataframe = pd.read_csv(os.path.join(data_dir, filename), engine='python')

    # add new assessment, domain, measures info to item names
    df = match_datadic_to_data(dataframe=dataframe)

    # add proprietry/free questionnaires to item names
    df_proprietary = pd.read_csv(os.path.join(data_dir, proprietary_filename))

    # add proprietary info to dataframe and assign NaN to Unknown
    df = df_proprietary.merge(df, on='datadic', how='outer')

    # remove rows that don't have any questions or keys
    conditional = (df['questions'].isna()) & (df['keys'].isna())
    df = df[~conditional]

    # fix domain name (missing domain in `Assessment_List_Jan2019.xlsx`)
    df = fix_domain(dataframe=df)

    # identify questions that contain total scores
    df = identify_total_scores(dataframe=df)

    # identify questions that are subheadings
    df = identify_subheadings(dataframe=df)

    # identify questions that are preambles
    df = identify_preamble(dataframe=df)

    # save out new file
    df.to_csv(os.path.join(data_dir, outname), index=False)


def add_demographics(
    dataframe, 
    fpath,
    participant_id='Identifiers',
    demos_to_add=['Sex', 'Age_round','PreInt_Demos_Fam,Child_Race_cat', 'PreInt_Demos_Fam,Child_Ethnicity_cat']
    ):
    """ add demographics to data files

    Args:
        dataframe (pd dataframe): dataframe to merge
        fpath (str): fullpath to `Clinical_Diagnosis_Demographics.csv`
        participant_id (str): column name of participant identifiers
        demos_to_add (list): list of columns to merge
    Returns:
        df (pd dataframe): merged dataframe
    """
    import itertools

    # read in clinical diagnosis demographics
    df = pd.read_csv(fpath, engine='python')

    cols_to_merge = [[participant_id], demos_to_add]
    cols_to_merge = list(itertools.chain(*cols_to_merge))

    # merge demo columns with dataframe
    df_out = df[cols_to_merge].merge(dataframe, on='Identifiers')

    return df_out


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


def merge_measures(
    assessment='Child',
    domains='all',
    measures='all',
    participants=os.path.join(Defaults.PHENO_DIR, 'participants.csv')
    ):
    """read in data from `data/raw/phenotype/Assessment_List_Jan2019.xlsx`:
    `assessment` (e.g., Child, Teacher, Parent, Teacher), `domains` (e.g., Cognitive Testing), `measures` (e.g., Kaufman Brief Intelligence Test-II)
    Creates new files based on combination of `assessment`, `domains`, and `measures`. Filters by `participants`

    Args: 
        assessment (str): options: 'Child', 'Parent', 'Teacher'. Default is 'Child'
        domains (list of str or 'all'): exhaustive list, find options here: `data/raw/phenotype/Assessment_List_Jan2019.xlsx`. Default is 'all'. If `domains` is 'all', all domains are loaded
        measures (list of str or 'all'): exhaustive list, find options here: `data/raw/phenotype/Assessment_List_Jan2019.xlsx`. Default is 'all'. If `measures` is 'all', all measures are loaded for `domains`
        participants (str): fullpath to `participants.csv`. Default is `../data/raw/phenotype/participants.csv`. Column `Identifiers` must be present.
    Returns:
        df_all (pd dataframe)
    """
    import re

    # check input args - `domains` and `measures` must be list or None
    if (isinstance(domains, str)) and ('all' in domains):
        domains = [domains]
    if (isinstance(measures, str)) and ('all' in measures):
        measures = [measures]

    # get directory
    fdir = os.path.join(Defaults.PHENO_DIR, f'{assessment}_Measures') # join with '_'

    # get domains
    if None in domains:
        domain_dir = [fdir]
    elif 'all' in domains:
        domain_dir = glob.glob(os.path.join(fdir, '*'))
    else:
        domain_dir = [os.path.join(fdir, '_'.join(re.split(r'_|,|/| ', d))) for d in domains]

    # `domain_dir` and `measure_dir` are the same for Teacher and Clinical measures
    if 'all' in measures and assessment in ['Teacher', 'Clinical']:
        measure_dir = domain_dir # these are the same for both Teacher and Clincial
        domain_dir = [1]

    # load in participants
    identifiers = pd.read_csv(participants, engine='python')['Identifiers']

    # loop over domains
    df_all = pd.DataFrame({'Identifiers': identifiers})
    df_participants = pd.DataFrame({'Identifiers': identifiers})
    for domain in domain_dir:

        # get measures
        if 'all' in measures and assessment in ['Parent', 'Child']:
            measure_dir = glob.glob(os.path.join(domain, '*'))
        elif 'all' not in measures:
            measure_dir = [os.path.join(domain, '_'.join(re.split(r'_|,|/| ', m)) + '.csv') for m in measures] # join with '_'

        # loop over measures
        for measure in measure_dir:
            
            # only read in files that exist
            if os.path.isfile(measure):
                df = pd.read_csv(measure, engine='python')
                df = df.loc[:,~df.columns.duplicated()].copy() # drop duplicate columns
                df = df.drop_duplicates() # drop duplicate rows
                # no min participants required
                df_all = df_all.merge(df, on="Identifiers", how='outer')

    # make sure only Identifiers from `df_participants` are going into dataframe
    for idx in df_all.index:
        row = df_all.loc[idx, 'Identifiers']
        if row in df_participants['Identifiers'].tolist():
            df_all.loc[idx, 'present'] = True
    df_all = df_all[df_all['present']==True]

    # drop NaN
    df_all = df_all.replace(' ', np.float("NaN")).fillna(np.float("NaN")).dropna(how='all', axis=1)
    df_all = df_all.dropna(how='all', axis=0)

    return df_all


def make_data_files(
    items_fpath,
    clinical_fpath,
    out_dir,
    participant_id='Identifiers',
    ):
    """ make data files and save in `Defaults.INTERIM_FEATURE_DIR`

    Args:   
        items_fpath (str): fullpath to `item-names-cleaned.csv`
        clinical_fpath (str): fullpath to `Clinical_Diagnoses_Demographics.csv`
        out_dir (str): directory where output files will be saved
        participant_id (str): column name of participant identifiers. default is 'Identifiers'

    Returns:
        saves out interim data files to `Defaults.INTERIM_FEATURE_DIR`
    """
    # make data files
    df_all = pd.DataFrame()
    for assessment in ['Child', 'Parent', 'Teacher']:
        # make data files
        df = merge_measures(
                assessment=assessment,
                domains='all',
                measures='all'
                )

        # add demographic information
        df_demos = add_demographics(dataframe=df, fpath=clinical_fpath)

        # do some additional cleaning
        df_clean, cols = remove_mixed_type_columns(dataframe=df_demos)

        # save assessment data to file
        df_clean.to_csv(os.path.join(out_dir, f'{assessment}-features-raw.csv'), index=False)

        # concatenate dataframes
        df_all = pd.concat([df_all, df_clean])

        # load `item-names-cleaned.csv`
        df_items = pd.read_csv(items_fpath, engine='python')
        
        # optionally filter datafiles
        filter_cols = ['Total_Scores', 'Free_Assessments', 'Proprietary_Assessments', 'Not_Total_Scores'] 

        # filter dataframes based on cols in `item-names-cleaned.csv`
        for filter_col in filter_cols:

            # initialize new dataframe
            features_filtered = pd.DataFrame()

            list_of_cols = list(df_items[df_items[filter_col]==True]['col_name'].dropna().unique())

            # add Identifiers to features
            list_of_cols.insert(0, 'Identifiers')

            # loop over columns to filter
            for col in list_of_cols:
                if col in df_clean.columns:
                    features_filtered.loc[:, col] = df[col]
        
            # save filtered data to file
            features_filtered.to_csv(os.path.join(out_dir, f'{assessment}-features-{filter_col}-raw.csv'), index=False)
    
    # save all assessments to file
    df_all.to_csv(os.path.join(out_dir, f'all-features-raw.csv'), index=False)
    

def run():

    # do some minimal preprocessing on the files
    separate_main_assessment_file_into_csvs()
    remove_redundant_identifiers_from_assessments()
    print('separated main assessment file into csvs and removed redundant identifiers from assessments', flush=True)

    # make `Clinical_Diagnosis_Demographics.csv` file
    make_clinical_summary_file(
        filename='Diagnosis_ClinicianConsensus.csv', 
        outname='Clinical_Diagnosis_Demographics.csv',
        data_dir=os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures')
        )
    print('created new clinical diagnosis demographics file', flush=True)

    # get all participant diagnoses
    get_all_participant_diagnoses(
        fpath=os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Clinical_Diagnosis_Demographics.csv'),
        outname='all_participant_diagnoses.csv',
        data_dir=Defaults.INTERIM_FEATURES_DIR
        )
    print('created participant diagnosis file', flush=True)

    # make `Free_Assessments_HBN_new.csv` file
    make_new_proprietary_assessments_file(
        filename='Free_Assessments_HBN.xlsx', 
        outname='Free_Assessments_HBN_new.csv',
        data_dir=Defaults.PHENO_DIR, 
        )
    print('created proprietary assessments file', flush=True)

    # make items file
    make_items_file(
        filename='item-names.csv', 
        proprietary_filename='Free_Assessments_HBN_new.csv',
        outname='item-names-cleaned.csv',
        data_dir=Defaults.PHENO_DIR,
    )
    print('created new item names file', flush=True)

    # make parent files
    make_data_files(
        items_fpath=os.path.join(Defaults.PHENO_DIR, 'item-names-cleaned.csv'),
        clinical_fpath=os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Clinical_Diagnosis_Demographics.csv'),
        out_dir=Defaults.INTERIM_FEATURES_DIR
        )
    print('created new data files', flush=True)

if __name__ == "__main__":
    run()
    

    
