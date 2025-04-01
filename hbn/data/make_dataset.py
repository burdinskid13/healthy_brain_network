import os
import numpy as np
import pandas as pd

import os
from hbn.constants import Defaults

import warnings
warnings.filterwarnings("ignore")

def cleanup_diagnosis_cols(df):
    # do some clean up
    df.columns = df.columns.str.replace('Diagnosis_ClinicianConsensus,', '')

    # replace NaN
    for num in range(1,11):
        # num
        num_str = str(num).zfill(2)
        
        # Replace "NaN" diagnosis with 'No Diagnosis Given: No Reason Given'
        df.loc[df[f'DX_{num_str}']==' ',f'DX_{num_str}'] = 'No Diagnosis Given: No Reason Given'
    df.loc[df[f'DX_{num_str}_Cat'].isna(),f'DX_{num_str}_Cat'] = 'No Diagnosis Given: No Reason Given'

    return df


def exclude_diagnoses(df, dx_to_exclude):
    # we're excluding diagnoses that have the following labels: RuleOut, Rem (remission), PRem (partial remission), RC (requires confirmation)
    # see full list here: https://docs.google.com/spreadsheets/d/1si0JDiI0rELnyQAGAaoO3lbERRfcKQ5X/edit?usp=sharing&ouid=115304373382106482578&rtpof=true&sd=true
    # we're doing this by setting these diagnoses to NaN
    for dx in dx_to_exclude:
        for num in np.arange(1,11):
            for col in [f'DX_{num:02d}', f'DX_{num:02d}_Cat', f'DX_{num:02d}_Cat_new']:
                df.loc[df[f'DX_{num:02d}_{dx}']==1, col] = np.nan

    # deal with missing values and NaN
    df = df.replace(' ', float("NaN")).fillna(float("NaN")).dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)

    return df


def melt_dx(df, cols_to_keep=['Identifiers']):
    dx_cols = [f'DX_{num:02d}' for num in np.arange(1,11)]
    dx_subtype = pd.melt(df, id_vars=cols_to_keep, value_vars=dx_cols, var_name='DX_Subtype', value_name='DX_Subtype_Name')

    dx_cols = [f'DX_{num:02d}_Cat_new' for num in np.arange(1,11)]
    dx_cat = pd.melt(df, id_vars=cols_to_keep, value_vars=dx_cols, var_name='DX_Cat', value_name='DX_Cat_Name')

    dx_concat = pd.concat([dx_cat, dx_subtype[['DX_Subtype', 'DX_Subtype_Name']]], axis=1)
    dx_concat = dx_concat.loc[:, ~dx_concat.columns.duplicated()].copy()

    return dx_concat


def get_all_diagnoses(dataframe):
    """Get all diagnoses from dataframe

    Args:
        dataframe (pd dataframe): must contain cols `DX_{num_str}_Cat_new` and `DX_{num_str}`. 
    Returns: 
        diagnoses_all (dict): dict of all diagnoses and subtypes
    """
    import numpy as np


    diagnoses_all = []; subtypes_all = []
    for num in range(1,11):
        
        num_str = str(num).zfill(2)

        # get categories
        diagnoses = dataframe[f'DX_{num_str}_Cat_new'].unique()
        subtypes = dataframe[f'DX_{num_str}'].unique()

        diagnoses_all.extend(diagnoses)
        subtypes_all.extend(subtypes)
    
    data_dict = {'diagnosis': np.unique(np.array(diagnoses_all)), 'subtype': np.unique(np.array(subtypes_all))}

    return data_dict


def filter_dataframe(dataframe, column='diagnosis', value='ADHD'):
    """ Return participant identifiers for either `diagnosis` (e.g., ADHD) or `subtype` ('ADHD-Combined Type')

    Args:
        dataframe (pd dataframe): must contain columns `DX_{num_str}_Cat_new` and `DX_{num_str}`.
        column (str): must be either 'diagnosis' or 'subtype'
        value (str): e.g. 'ADHD'    
    Returns:
        dataframe_filtered (pd dataframe): contains columns `Identifiers`
    """

    subjs_all = []
    for num in range(1, 11):
        num_str = str(num).zfill(2)

        # diagnosis or subtype?
        if column=='diagnosis':
            col = f'DX_{num_str}_Cat_new'
        elif column=='subtype':
            col = f'DX_{num_str}'

        # Filtering dataframe for diagnosis
        subjs = dataframe[(dataframe[col]==value) &
                (dataframe[f'DX_{num_str}_RuleOut']!=1) &
                (dataframe[f'DX_{num_str}_Rem']!=1)
                   ]['Identifiers']
        subjs_all.extend(subjs)

    # return select participants
    dataframe_filtered = dataframe[dataframe['Identifiers'].isin(subjs_all)]

    # figure out if participants have more than one diagnosis
    dataframe_filtered.loc[dataframe_filtered['comorbidities']>0, 'only_diagnosis'] = False
    dataframe_filtered.loc[dataframe_filtered['comorbidities']==0, 'only_diagnosis'] = True

    return dataframe_filtered


def define_new_categories(dataframe):
    """define new disorder categories using labels from `DX_<num>_Cat`

    Args:
        dataframe (pd dataframe):  Must contain columns `DX_{num_str}_Cat_new` and `DX_{num_str}`.
    Returns:
        dataframe (pd dataframe): contains columns `DX_{num_str}_Cat_new`
    """
    import pandas as pd

    def new_categories(x,y):
        adhd_list = ['ADHD', 'Attention-Deficit']
        autism_list = ['Autism']
        learning_list = ['Specific Learning Disorder with Impairment in Reading']
        if isinstance(x, str):
            adhd = any(map(x.__contains__, adhd_list))
            autism = any(map(x.__contains__, autism_list))
            learning = any(map(x.__contains__, learning_list))
            if adhd:
                return 'ADHD'
            elif autism:
                return 'Autism Spectrum Disorder'
            elif learning:
                return 'Specific Learning Disorder with Impairment in Reading'
            else:
                return y

        
    dx_to_model = ['Anxiety Disorders', 'Autism Spectrum Disorder', 'ADHD', 'No Diagnosis Given: No Reason Given',
                'No Diagnosis Given', 'No Diagnosis Given: Incomplete Eval',
                'Specific Learning Disorder with Impairment in Reading']

    for num in range(1,11):
        
        num_str = str(num).zfill(2)
        
        ## divide neurodevelopmental disorders into other categories
        dataframe[f'DX_{num_str}_Cat_new'] = dataframe.apply(lambda x: new_categories(x[f'DX_{num_str}'], x[f'DX_{num_str}_Cat']), axis=1)

    return dataframe


def _remove_redundant_str(df):
    df['Identifiers'] = df['Identifiers'].str.strip(r',assessment|,,assessment|').str.extract(r'(\w+)', expand=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[~df['Identifiers'].isna()]
    return df


def _check_for_exact_match_csv(cell, target):
    # remap keys
    remap_keys = {'TRF_P': 'TRF_Pre',
        'ARI_SR': 'ARI_S',
        'Audit': 'AUDIT',
        'FLANKER': 'Flanker',
        'CELF5_Meta': 'CELF_Meta',
        'PhenX_SchoolRisk': 'PhenX_School',
        'PreInt_FamHx': 'PreInt_FamHx_RDC'
        }
    if target in remap_keys.keys():
        target = remap_keys[target]

    if isinstance(cell, str):
        parts = [part.strip() for part in cell.split(',')]  # Split and trim
        return target in parts
    else:
        return False #handles non string types.


def parse_csv_files( 
    data_dir,
    release='Release11_Apr2024', 
    assessment='HBN_Assessment_List_shared_Apr2024.xlsx',
    filter_col='Data shared R11',
    filter_val='Yes'
    ):
    """ parse csv files and sort into respective domain folders 

    assessment (str): full name of assessment list `HBN_Assessment_List_shared_Apr2024.xlsx`
    release (str): full name of release (e.g., Release11_Apr2024)
    data_dir (str): fullpath to top-level directory of raw data (e.g., `Defaults.PHENO_DIR`)
    """
    import os
    import glob
    import pandas as pd
    from pathlib import Path
    import shutil

    # open data dictionary file
    df_dict = pd.read_excel(os.path.join(data_dir, assessment))

    # filter assessment
    if filter_col is not None:
        df_dict = df_dict[df_dict[filter_col]==filter_val]

    # grab all csv files
    csv_files = glob.glob(os.path.join(data_dir, release, '*.csv'))

    # create domain folders
    domains = df_dict['Domain'].unique()
    for domain in domains:
        dest = os.path.join(data_dir, release, domain)
        os.makedirs(dest, exist_ok=True)

    # get LORIS abbreviations
    col_name = 'Abbreviation(s) LORIS'
    abbrev_loris = df_dict[col_name].unique()

    # assign each csv file to its respective domain folder
    for csv in csv_files:

        # read csv
        df = pd.read_csv(csv, engine='python')
        
        # remove redundant str
        df = _remove_redundant_str(df)

        # check that the abbrev (e.g., FGC) in column names matches the data dic
        abbrev = df.columns[1].split(',')[0]
        datadic = Path(csv).name.split('.')[0]

        # if column names are the same as csv file name, then change column names to be the same
        if abbrev!=datadic:
            df.columns = [col.replace(abbrev, datadic) for col in df.columns]
            print(f'{csv} file name is not the same as column names')
        
        # save csv to domain folder
        df_filter = df_dict[df_dict[col_name].apply(lambda x: _check_for_exact_match_csv(x, datadic))]
        if not df_filter.empty:
            domain = df_filter['Domain'].values[0]
            assessment = df_filter['Measure Participant'].values[0]
            outdir = os.path.join(data_dir, release, assessment, domain)
        else:
            outdir = os.path.join(data_dir, release, 'Unknown', 'Unknown')

        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, Path(csv).name)
        df.to_csv(outpath, index=False)


def _lookup_data_dict(csv, data_dir):
	"""find corresponding data dictionary xlsx for a given csv file

	Args: 
		csv (str): csv filename (e.g., `ACE.csv`)
		data_dir (str): fullpath where xlsx files are saved
	Returns: 
		xlsx (str): xlsx filename
	"""
	import os 
	from pathlib import Path

	csv = Path(csv).name.replace('.csv', '')

	fpath = os.path.join(data_dir, csv + '.xlsx')
	if os.path.exists(fpath):
		return fpath
	else:
		None


def _identify_scores_and_assign_col_type(df):
    """
    Identifies rows with "Scores" in the "Question" column and assigns "Total_Scores"
    to the "col_type" column in subsequent rows.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with the "col_type" column.
    """

    df['col_type'] = float(np.nan)  # Initialize the new column with None
    df['col_type'] = df['col_type'].astype(object)

    scores_found = False  # Flag to track if "Scores" has been found
    for index, row in df.iterrows():
        if "Scores" in str(row['Question']): #Handles potential nan values
            scores_found = True
        elif scores_found:
            df.loc[index, 'col_type'] = "Scores"

    return df


def _fix_symchck_dict(df):
    import re
    
    # Create a dictionary to store the values from 'Question' corresponding to 'CSC_*C' values in 'Variable Name'
    c_values = {}

    # Iterate through the DataFrame to populate the dictionary
    for index, row in df.iterrows():
        if pd.notna(row['Variable Name']) and re.match(r'CSC_.*C', row['Variable Name']):  # Check for NaN first
            c_values[row['Variable Name']] = row['Question']

    # Iterate through the DataFrame to populate NaN values
    for index, row in df.iterrows():
        if pd.notna(row['Variable Name']) and re.match(r'CSC_.*P', row['Variable Name']):  # Check for NaN first
            corresponding_c = row['Variable Name'].replace('P', 'C')
            if corresponding_c in c_values:
                df.loc[index, 'Question'] = c_values[corresponding_c]
    
    return df


def _check_exceptions(df, abbrev):
    """there's a mismatch between some data dictionaries (.xlsx) and data (.csv) files. 
    for example, 'CBCLPre,..' is coded in .csv 'CBCL_Pre.csv' but correct abbrev (as listed in .xlsx) is 'CBCL_Pre'
    this needs to be corrected and the correct mapping provided so that the data cols can be correctly indexed
    """
    import re

    merge_col = 'data_col_name_map'
    if abbrev=='CBCL_Pre':
        df['data_col_name_map'] = df['data_col_name'].str.replace('CBCLPre', abbrev).str.replace('CBCLpre', abbrev)
    elif abbrev=='BIA':
        df['data_col_name_map'] = df['data_col_name'].str.replace(f'{abbrev}_', '')
    elif abbrev=='Barratt':
        df['data_col_name_map'] = df['data_col_name'].str.replace('Barratt_Total_Edu', 'Total_Edu').str.replace( 'Barratt_Total_Occ', 'Occupation_Total')
    elif abbrev=='ICU_P':
        df['data_col_name_map'] = df['data_col_name'].str.replace('ICU_P', 'ICU').str.replace('ICU_Total', 'ICU_P_Total')
    elif abbrev=='DTS':
        df['data_col_name_map'] =  df['data_col_name'].str.replace('DTS_Total', 'DTS_total')
    elif abbrev=='CTOPP':
        df['data_col_name_map'] = df['data_col_name'] \
                                    .str.replace(r'CTOPP_(..)_Sum', r'\1_sum', regex=True) \
                                    .str.replace(r'CTOPP_(..)_comp', r'\1_composite', regex=True) \
                                    .str.replace(r'CTOPP_(..)_P', r'\1_percentile', regex=True) \
                                    .str.replace(r'CTOPP_(..)_S', r'\1_scaled', regex=True) \
                                    .str.replace(r'CTOPP_(..)_D', r'\1_desc', regex=True) \
                                    .str.replace(r'CTOPP_(..)_R', r'\1_raw', regex=True)
    elif abbrev=='RBS': 
        df['data_col_name_map'] = df['data_col_name'].str.replace('RBS_Score', 'Score')
    elif abbrev=='SDQ':
        df['data_col_name_map'] = df['data_col_name'].str.replace('SDQ_Conduct_Problems', 'Conduct_Problems_Total') \
                                                     .str.replace('SDQ_Difficulties_Total', 'Difficulties_Total').str.replace('SDQ_Emotional_Problems', 'Emotional_Problems_Total') \
                                                     .str.replace('SDQ_Externalizing', 'Externalising_Total').str.replace('SDQ_Generating_Impact', 'Generating_Impact_Total') \
                                                     .str.replace('SDQ_Hyperactivity', 'Hyperactivity_Total').str.replace('SDQ_Internalizing', 'Internalising_Total') \
                                                     .str.replace('SDQ_Peer_Problems', 'Peer_Problems_Total').str.replace('SDQ_Prosocial', 'Prosocial_Total') 
    elif abbrev=='YFAS_C':
        df['data_col_name_map'] = df['data_col_name'].str.replace('YFAS_C', 'YFAS')
    else:
        df['data_col_name_map'] = float("nan")
        merge_col = 'data_col_name'

    return df, merge_col


def make_items(data_dir, outname='item-names.csv'):
    import os
    import glob
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # grab all csv files
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)

    # grab all data dict files
    dict_files = glob.glob(os.path.join(data_dir, '*.xlsx'))

    # ignore any diagnosis files (those will not go in question-column lookup file)
    files_to_ignore = ['KSADS', 'Diagnosis', 'item-names']
    csv_files_updated = [csv for csv in csv_files if not any(key in Path(csv).name for key in files_to_ignore)]

    # loop over all csv files
    df_merged_all = pd.DataFrame()
    for csv in csv_files_updated:

        # grab csv file
        df_csv = pd.read_csv(csv, engine='python')

        # which assessment, domain - relies on structure of data organization
        domain = Path(csv).parent.name
        assessment = Path(csv).parent.parent.name

        # grab all columns after the first Identifiers col
        col_names = [col.split(',')[1] for col in df_csv.columns[1:]]

        # abbrev name
        col_names_abbrev = df_csv.columns[1].split(',')[0]

        df = pd.DataFrame()
        df['data_col_name'] = col_names

        # check if there are exceptions, mismatches between csv data col names and correct abbrev
        df, merge_col = _check_exceptions(df, col_names_abbrev)

        # look up data dictionary
        xlsx = _lookup_data_dict(csv, data_dir)

        if xlsx is not None:
            df_xlsx = pd.read_excel(xlsx, header=1)

            # 'SympChck' dictionary needs to be fixed
            if col_names_abbrev=='SympChck':
                df_xlsx = _fix_symchck_dict(df_xlsx)

            # delete Unnamed columns
            df_xlsx = df_xlsx.drop(columns=[col for col in df_xlsx.columns if "Unnamed" in col])

            # rename cols 
            keywords = ['Question', 'Scores', 'Item', 'Subtest']
            check = any(keyword in col for col in df_xlsx.columns.tolist() for keyword in keywords)
            if not check:
                df_xlsx = pd.read_excel(xlsx, header=2)

            col_names = ['Question', 'Variable Name', 'Variable Type', 'Value', 'Value Labels']

            current_cols = df_xlsx.columns.tolist()
            for idx, col in enumerate(col_names):
                current_cols[idx] = col
            df_xlsx.columns = current_cols
            df_xlsx = df_xlsx[col_names]

            # create new conditional col
            df_xlsx = _identify_scores_and_assign_col_type(df_xlsx)

            # get questions corresponding to column name
            df_merged = df_xlsx.merge(df, left_on='Variable Name', right_on=merge_col, how='outer')

            df_merged['abbrev'] = col_names_abbrev
            df_merged['domain'] = domain
            df_merged['assessment'] = assessment

            # figure out variable/col name is a question/subheading/administrative col/total score
            df_merged = _create_conditional_col(df_merged)

            # concat all dataframes
            df_merged_all = pd.concat([df_merged_all, df_merged])
        else:
            import logging
            logging.basicConfig(filename=os.path.join(data_dir, 'item-names-log.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info(f'there is no corresponding data dictionary for {Path(csv).name}')

    df_merged_all.to_csv(os.path.join(data_dir, outname), index=False)


def _create_conditional_col(df):
    """
    updates conditional column with multiple conditions. `col_type` should already be in `df`

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with the "col_type" column.
    """
    import numpy as np
    import pandas as pd
    
    if 'col_type' not in df.columns:
        df['col_type'] = float(np.nan)  # Initialize the new column with None
        df['col_type'] = df['col_type'].astype(object)

    # Condition 1: subheading
    condition1 = (df['Question'].notna()) & (df['Variable Name'].isna()) & (df['Variable Type'].isna())
    df.loc[condition1, 'col_type'] = 'Subheading'

    # Condition 2: Question
    condition2 = (df['col_type'].isna() & (df['Question'].notna()))
    df.loc[condition2, 'col_type'] = 'Question'

    # Condition 3: Total Scores
    condition2 = (df['col_type'].isna() & (df['data_col_name'].str.contains(r'Score|Total', case=False, na=False)))
    df.loc[condition2, 'col_type'] = 'Scores'

    # Condition 4: Unknown
    condition3 = (df['data_col_name'].notna()) & (df['Question'].isna())
    df.loc[condition3, 'col_type'] = 'Unknown'

    return df


def get_demographics(
    df,
    child_dir,
    parent_dir,
    ):
    """add demographics to existing dataframe, merging on participant id `Identifiers`

    Args: 
        df (pd dataframe):  Must contain columns `Identifiers`
        child_demos (pd dataframe): directory where child demographic files are saved 
        parent_demos (pd dataframe): directory where parent demographic files are saved
    Returns:
        merged_df (pd dataframe): returns `dataframe` with demographic columns for each `Identifiers`
    """
    import os
    import pandas as pd
    from functools import reduce

    # assign directories
    child_demos_dir = os.path.join(child_dir, 'Demographics')
    parent_demos_dir = os.path.join(parent_dir, 'Demographics')
    parent_interview_dir = os.path.join(parent_dir, 'Emotional and Psychological Function')

    # READ BASIC DEMOGRAPHICS FILES
    df_demo = pd.read_csv(os.path.join(child_demos_dir, 'Basic_Demos.csv'))[['Identifiers', 'Basic_Demos,Sex', 'Basic_Demos,Age', 'Basic_Demos,Study_Site']]
    df_race = pd.read_csv(os.path.join(parent_interview_dir, 'PreInt_Demos_Fam.csv'))[['Identifiers', 'PreInt_Demos_Fam,Child_Race', 'PreInt_Demos_Fam,Child_Ethnicity']]
    df_edu = pd.read_csv(os.path.join(parent_demos_dir, 'Barratt.csv'))[['Identifiers', 'Barratt,Barratt_Total_Edu']]
    df_ses = pd.read_csv(os.path.join(parent_demos_dir, 'FSQ.csv'))[['Identifiers', 'FSQ,FSQ_04']]
    df_dev = pd.read_csv(os.path.join(parent_interview_dir, 'PreInt_DevHx.csv'))[['Identifiers', 'PreInt_DevHx,puberty']]

    # assign new variables
    df_demo['sex'] = df_demo['Basic_Demos,Sex'].map({0: 'male', 1: 'female'})
    df_demo['age_round'] = df_demo['Basic_Demos,Age'].round()
    df_demo['study_site'] = df_demo['Basic_Demos,Study_Site']
    df_dev['puberty'] = df_dev['PreInt_DevHx,puberty'].map({0: 'pre', 1: 'post'})
    df_ses['household_income'] = df_ses['FSQ,FSQ_04'].map(_remap_ses())
    df_edu.loc[df_edu['Barratt,Barratt_Total_Edu']>=18, 'education'] = 'College Degree or Higher'
    df_edu.loc[df_edu['Barratt,Barratt_Total_Edu']<18, 'education'] = 'Less than College Degree'
    df_race['Identifiers'] = df_race['Identifiers'].str.replace(',assessment', '')
    df_race['race'] = df_race['PreInt_Demos_Fam,Child_Race'].fillna(10).apply(lambda x: _remap_race(x)) # fill NaN values with "Unknown"
    df_race['ethnicity'] = df_race['PreInt_Demos_Fam,Child_Ethnicity'].fillna(3).apply(lambda x: _remap_ethnicity(x)) # fill NaN values with "Unknown"

    # merge dataframes
    dfs = [df_demo, df_dev, df_ses, df_edu, df_race.iloc[1:]]  # List of DataFrames
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Identifiers', how='outer'), dfs)

    cols_to_keep = ['Identifiers', 'sex', 'age_round', 'study_site', 'puberty', 'household_income', 'education', 'race', 'ethnicity']
    merged_df = merged_df[cols_to_keep]

    # add development stage
    merged_df = _add_development_stage(df=merged_df)

    return merged_df


def _add_comorbidities(df):
    # get comorbidities
    dx_counts = df.groupby('Identifiers')['DX_Cat_Name'].apply(lambda x: x.nunique()-1).reset_index(name='comorbidities')
    df = dx_counts.merge(df, on='Identifiers')

    return df


def _add_development_stage(df):
    df.loc[df['age_round'].isin([6,7,8]), 'reading_stage'] = 'Early'
    df.loc[df['age_round'].isin([9,10]), 'reading_stage'] = 'Emerging'
    df.loc[df['age_round'].isin([11,12,13,14,15,16,17,18]), 'reading_stage'] = 'Expert'

    return df


def _add_group_reading(df, reading='Specific Learning Disorder with Impairment in Reading'):
    import pandas as pd
    from scipy import stats as sp

    # loop over participant groups and assign new groups- ORDER OF STATEMENTS MATTERS
    df_all = pd.DataFrame()
    for _, group in df.groupby('Identifiers'):
        dx = group['DX_Cat_Name'].values
        if 'No Diagnosis Given' in dx:
            group['DX_Reading'] = 'No Diagnosis Given'
        elif (reading in dx) and (sum(group['comorbidities'])==0):
            group['DX_Reading'] = 'reading_no_comorbidities'
        elif ('ADHD' in dx) and (reading not in dx):
            group['DX_Reading'] = 'adhd_no_reading' 
        elif reading in dx and (sum(group['comorbidities'])>0):
            group['DX_Reading'] = 'reading_all_comorbidities' 
        else:
            group['DX_Reading'] = 'other_diagnoses'
        df_all = pd.concat([df_all, group]) 

    return df_all


def _add_group_adhd(df, adhd='ADHD'):
    import pandas as pd
    from scipy import stats as sp

    # loop over participant groups and assign new groups- ORDER OF STATEMENTS MATTERS
    df_all = pd.DataFrame()
    for _, group in df.groupby('Identifiers'):
        dx = group['DX_Cat_Name'].values
        if 'No Diagnosis Given' in dx:
            group['DX_ADHD'] = 'No Diagnosis Given'
        elif (adhd in dx) and (sum(group['comorbidities'])==0):
            group['DX_ADHD'] = 'adhd_no_comorbidities'
        elif ('ADHD' in dx) and (sum(group['comorbidities'])>0):
            group['DX_ADHD'] = 'adhd_all_comorbidities' 
        else:
            group['DX_ADHD'] = 'other_diagnoses'
        df_all = pd.concat([df_all, group]) 

    return df_all


def _add_group_depression(df, depression='Depressive Disorders'):
    import pandas as pd
    from scipy import stats as sp

    # get rows that contain 'ADHD'
    mask = df['DX_Cat_Name'].str.contains('ADHD', case=False)
    mask = mask.fillna(False)

    # assign new groups
    df.loc[mask, 'DX_Cat_Name'] = df.loc[mask, 'DX_Subtype_Name']

    # loop over participant groups and assign new groups- ORDER OF STATEMENTS MATTERS
    df_all = pd.DataFrame()
    for _, group in df.groupby('Identifiers'):
        dx = group['DX_Cat_Name'].values
        if 'No Diagnosis Given' in dx:
            group['DX_Depression'] = 'No Diagnosis Given'
        elif (depression in dx) and not (any("ADHD" in str(item) for item in dx)):
            group['DX_Depression'] = 'Depression (no ADHD)'
        elif (depression not in dx) and ('ADHD-Combined Type' in dx):
            group['DX_Depression'] = 'ADHD-Combined Type (no Depression)'
        elif (depression not in dx) and ('ADHD-Inattentive Type' in dx):
            group['DX_Depression'] = 'ADHD-Inattentive Type (no Depression)'
        else:
            group['DX_Depression'] = 'other_diagnoses'
        df_all = pd.concat([df_all, group]) 

    return df_all


def _add_group_anxiety(df, anxiety='Anxiety Disorders'):
    import pandas as pd
    from scipy import stats as sp

    # get rows that contain 'ADHD'
    mask = df['DX_Cat_Name'].str.contains('ADHD', case=False)
    mask = mask.fillna(False)

    # assign new groups
    df.loc[mask, 'DX_Cat_Name'] = df.loc[mask, 'DX_Subtype_Name']

    # loop over participant groups and assign new groups- ORDER OF STATEMENTS MATTERS
    df_all = pd.DataFrame()
    for _, group in df.groupby('Identifiers'):
        dx = group['DX_Cat_Name'].values
        if 'No Diagnosis Given' in dx:
            group['DX_Anxiety'] = 'No Diagnosis Given'
        elif (anxiety in dx) and not (any("ADHD" in str(item) for item in dx)):
            group['DX_Anxiety'] = 'Anxiety (no ADHD)'
        elif (anxiety not in dx) and ('ADHD-Combined Type' in dx):
            group['DX_Anxiety'] = 'ADHD-Combined Type (no Anxiety)'
        elif (anxiety not in dx) and ('ADHD-Inattentive Type' in dx):
            group['DX_Anxiety'] = 'ADHD-Inattentive Type (no Anxiety)'
        else:
            group['DX_Anxiety'] = 'other_diagnoses'
        df_all = pd.concat([df_all, group]) 

    return df_all


def add_diagnosis_groups(df):
    # add comorbidities
    df = _add_comorbidities(df=df)

    # add reading-specific cols
    df = _add_group_reading(df=df, reading='Specific Learning Disorder with Impairment in Reading')

    # add adhd-specific cols
    df = _add_group_adhd(df=df, adhd='ADHD')

    # add depression-specific cols
    df = _add_group_depression(df=df, depression='Depressive Disorders')

    # add anxiety-specific cols
    df = _add_group_anxiety(df=df, anxiety='Anxiety Disorders')

    return df


def _remap_ses():
    return {
        0: "<$10,000",
        1: "$10,000 - $19,999",
        2: "$20,000 - $29,999",
        3: "$30,000 - $39,999",
        4: "$40,000 - $49,999",
        5: "$50,000 - $59,999",
        6: "$60,000 - $69,999",
        7: "$70,000 - $79,999",
        8: "$80,000 - $89,999",
        9: "$90,000 - $99,999",
        10: "$100,000 - $149,999",
        11: "$150,000 or more"
        }

    return df_out


def _remap_race(x):
    race_dict = {
        0: "White/Caucasian",
        1:"Black/African American",
        2:"Hispanic",
        3:"Asian",
        4:"Asian",
        5:"Native American",
        6:"Native American",
        7:"Native Hawaiian/Other Pacific Islander",
        8:"Two or more races",
        9:"Unknown",
        10:"Unknown",
        11:"Unknown"
        }
    return race_dict[x]


def _remap_ethnicity(x):
    ethnicity_dict = {
        0: "White/Caucasian",
        1: "Hispanic or Latino",
        2: "Unknown",
        3: "Unknown",
        }
    return ethnicity_dict[x]


