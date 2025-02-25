import os
import numpy as np
import pandas as pd

import os
from hbn.constants import Defaults


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


def get_domains(assessment='Child Measures'):
    """get domains for `assessment`

    Args:
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'
    Returns:
        list of str: list of domains
    """
    fname = '_'.join(assessment.split())

    # master info file
    fpath = os.path.join(Defaults.PHENO_DIR, f'Assessment_List_Jan2019_{fname}.csv')

    if not os.path.isfile(fpath):
        assessment_list(assessment=assessment)
    
    # read in corrected assessment list
    info = pd.read_csv(fpath)

    if 'Domain' in info.columns:
        domains = info['Domain'].unique().tolist()

        return {assessment: domains + ['all']}
    else:
        return {assessment: None}


def get_measures(assessment='Child Measures', domain='Cognitive Testing'):
    """get measures for `assessment` and `domain`. See `Assessment_List_2019.xlsx` for `assessment` and `domain`

    Args:
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'
        domain (str or None): specific for each assessment. if 'all', then measures for all domains are returned.
    Returns:
        list of str: list of domains
    """
    fname = '_'.join(assessment.split())

    # info file
    fpath = os.path.join(Defaults.PHENO_DIR, f'Assessment_List_Jan2019_{fname}.csv')

    if not os.path.isfile(fpath):
        assessment_list(assessment=assessment)
    
    # read in corrected assessment list
    info = pd.read_csv(fpath)

    # return measures if both domain and measures are present
    if sum(info.columns.isin(['Domain', 'Measure']))==2:
        if domain != 'all':
            measures = info[info['Domain']==domain]['Measure'].tolist()
        else:
            measures = []
            for name, group in info.groupby('Domain'):
                measures.extend(group['Measure'].tolist())
    else:
        measures = info['Measure']

    return {domain: measures}


def get_abbrevs(assessment='Child Measures', measure='Grooved Pegboard'):
    """get data dictionaries (.xlsx) for each `measure`

    Args:
        measure (str): default is 'Grooved Pegboard'
    Returns:
        list of dicts
    """
    abbrev = 'Abbreviation(s) LORIS'

    def _check_exceptions(assessment, measure):
        # where there is a mismatch between the abbrev in the assessment list and the measure csv
        exceptions = {
            'Child_Measures': 
            {
            'NIH Toolbox': ['NIH_final', 'NIH_Scores'],
            'Temporal Discounting Task': ['temp_disc_final'],
            'Kiddie Schedule for Affective Disorders and Schizophrenia': ['KSADS_C'],
            'Body Composition': ['bia_final'],
            'Alcohol Use Disorders Identification Test ': ['Audit'],
            'Food Frequency Questionnaire-Screening Form': ['FFQ_final']
            },
            'Parent_Measures': 
            {
            'Kiddie Schedule for Affective Disorders and Schizophrenia': ['KSADS_P']
            }
        }
        try:
            return exceptions[assessment][measure]
        except:
            return None

    assessment = '_'.join(assessment.split(' '))
    
    # loop over assessments
    if assessment=='Teacher_Measures':
        abbrev = 'Abbreviation'
    fpath = os.path.join(Defaults.PHENO_DIR, f'Assessment_List_Jan2019_{assessment}.csv')

    if not os.path.isfile(fpath):
        assessment_list(assessment=assessment)
    
    # read in corrected assessment list
    info = pd.read_csv(fpath)

    # loop over measures
    match = info[info['Measure']==measure]
    if not match.empty:
        abbrevs = match[abbrev].tolist()
        if measure=='Clinical Evaluation of Language Fundamentals':
            abbrevs = ['CELF_Full_5to8', 'CELF_Full_9to21']
        elif measure=='Treadmill Test':
            abbrevs = ['Fitness_Aerobic', 'Fitness_Endurance']
        elif measure=='Intake Interview':
            abbrevs = ['PreInt_Demos_Fam', 'PreInt_Demos_Home', 'PreInt_DevHx', 'PreInt_EduHx', 'PreInt_Lang', 'PreInt_TxHx']
        else:
            abbrevs = abbrevs[0].split(', ')
        # check exceptions
        exception = _check_exceptions(assessment, measure)
        if exception is not None:
            abbrevs = exception
        return abbrevs


def get_datadic(abbrev='NIH_final', release='Release9_DataDic_Nov2020'):
    
    # datadic file
    fpath = os.path.join(Defaults.PHENO_DIR, release, f'{abbrev}.xlsx')

    def _check_exceptions(abbrev):
        exceptions = {
            'NIH_final': 'NIH_Full',
            'temp_disc_final': 'Temp_Disc',
            'bia_final': 'BIA',
            'Basic_Demos': 'BasicDemos',
            'TRF_Pre': 'TRF_P',
            'Audit': 'AUDIT'
            }
        try:
            return exceptions[abbrev]
        except:
            return None

    # read excel
    if not os.path.isfile(fpath):
        abbrev = _check_exceptions(abbrev)

    return abbrev


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


def separate_main_assessment_file_into_csvs(
        assessment, 
        fpath=os.path.join(Defaults.PHENO_DIR, 'Assessment_List_Jan2019.xlsx'), 
        data_dir=Defaults.PHENO_DIR
        ):
    """correct assessment list, update `domain` for each `measure`

    Args:
        assessment (str): e.g., 'Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'
        assessment_file (str): full path to `Assessment_List_Jan2019.xlsx`
        data_dir (str): directory where output will be saved. Default is `PHENO_DIR`
    Returns: 
        info (pd dataframe), domain (str)
    """
    # parse `assessment` sheets from `fpath`
    xls = pd.ExcelFile(fpath, engine='openpyxl');

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


def remove_redundant_identifiers_from_questionnaires(data_dir):
    """ some basic clean up on assessment files, remove redundant identifiers from questionnaire csvs

    Args:
        data_dir (str): fullpath to directory where csv files for assessment are saved
    """
    # get all csv files within assessment directory
    fpaths = glob.glob(f'{data_dir}/*/*.csv')
    # loop over files
    for fpath in fpaths:
        df = pd.read_csv(fpath, engine='python')
        df['Identifiers'] = df['Identifiers'].str.strip(r',assessment|,,assessment|').str.extract(r'(\w+)', expand=False)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df[~df['Identifiers'].isna()]
        df.to_csv(fpath, index=False)


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


def match_datadic_to_data(dataframe):
    """add new columns to item names dataframe
    """
    import os
    from hbn import io
    import math
    import glob
    from collections import defaultdict
    from hbn.constants import Defaults
    from hbn.specs import make_specs


    # grab all feature files and make dictionary from abbrevs and datadic args
    fdir = os.path.join(Defaults.FEATURE_DIR, 'basic_demographics')
    parent_spec = os.path.join(fdir, 'features-parent_spec.json')
    if not os.path.isdir(fdir):
        make_specs.make_feature_specs(parent_spec, out_dir=fdir) # make feature spec files if they don't exist
    feature_specs = glob.glob(os.path.join(fdir, '*features*'))

    # initializing dict with lists
    new_dict = defaultdict(list)

    # loop over feature specs
    for spec in feature_specs:
        if 'features-parent_spec' not in spec:
            info = io.read_json(spec)

            dict = {info['datadic']: [info['abbrevs'], info['assessment'], info['domains'], info['measures']]}
        for k,v in dict.items():
            new_dict[k].append(v)

    # assign column names so that data can be indexed correctly
    for index in dataframe.index:
        key = dataframe.loc[index, 'datadic']
        is_str = type(dataframe.loc[index, 'keys']) is str
        if key in new_dict and is_str:
            dataframe.loc[index, 'col_name'] = new_dict[key][0][0] + ',' + dataframe.loc[index, 'keys']
            dataframe.loc[index, 'assessment'] = new_dict[key][0][1]
            dataframe.loc[index, 'domains'] = new_dict[key][0][2]
            dataframe.loc[index, 'measures'] = new_dict[key][0][3]

    return dataframe


def fix_domain(dataframe):
    measures_to_change = ['SympChck', 'ICU_P', 'ARI_P', 'SRS_Pre', 'SRS', 'RBS', 'SDQ', 'WHODAS_P', 'SAS', 
                'CIS_P', 'SCQ', 'ASSQ', 'SWAN','ESWAN','SCARED_P','MFQ_P', 'CBCL', 'CBCL_Pre']
    for abbrev in dataframe['datadic'].unique():
        if abbrev in measures_to_change:
            dataframe.loc[(dataframe["datadic"]==abbrev) & (~dataframe['keys'].isna()), "domains"] = 'Questionnaire_Measures_of_Emotional_and_Cognitive_Status'

    return dataframe


def identify_total_scores(dataframe):

    overall_scores = ['Total', 'Raw Score', 'T-Score', 'T Score', 'Standard']

    # filter dataframe
    questions_to_filter = dataframe[dataframe['questions'].str.contains('|'.join(overall_scores))==True]['questions'].unique()

    # assign new column to identify whether qustion is a total score or not
    dataframe['Total_Scores'] = dataframe['questions'].isin(questions_to_filter)
    dataframe.loc[dataframe['Total_Scores']==False, 'Not_Total_Scores'] = True

    return dataframe


def identify_preamble(dataframe):

    # anything is a preamble if 'keys' is empty and is not a subheading
    conditional = (dataframe['keys'].isna()) & (dataframe['Subheadings']==False) 

    # filter dataframe
    questions_to_filter = dataframe[conditional]['questions'].unique()

    dataframe['Preamble'] = dataframe['questions'].isin(questions_to_filter)

    # figure out which preambles are actually subheadings and reassign:
    subheadings = ['Other (1)', 'Other (2)', 'Suicidal ideation', 'Suicidal behavior', 'Social Anxiety',
                'Panic Disorder', 'Positive Behavior Scale Score']

    dataframe.loc[dataframe['questions'].isin(subheadings), 'Subheadings'] = True
    dataframe.loc[dataframe['questions'].isin(subheadings), 'Preamble'] = False

    return dataframe


def identify_subheadings(dataframe):

    # find keys with exact match: "Scores", "Scale Score", "Scoring"
    # and find keys that contain "Scales"
    conditional = (dataframe['questions']=='Scores') | (dataframe['questions'].str.contains('Scale Scores')) | (dataframe['questions']=='Scoring') |  (dataframe['questions'].str.contains('Scales')) & (dataframe['keys'].isna())
    
    # filter dataframe
    questions_to_filter = dataframe[conditional]['questions'].unique()

    # assign new column to identify whether qustion is a total score or not
    dataframe['Subheadings'] = dataframe['questions'].isin(questions_to_filter)

    return dataframe


def add_demographics(dataframe):
    """add demographics to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe):  Must contain columns `Identifiers`.
    Returns:
        dataframe (pd dataframe): returns `dataframe` with additional demographic columns
    """
    # READ BASIC DEMOGRAPHICS
    df_demo = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Parent_Measures/Demographic_Questionnaire_Measures/Basic_Demos.csv'))
    df_demo.columns = df_demo.columns.str.replace('Basic_Demos,','')
    df_demo['Sex'] = df_demo['Sex'].map({0: 'male', 1: 'female'})

    df_merged = df_demo[['Identifiers', 'Age', 'Sex', 'Enroll_Year']].merge(dataframe, on='Identifiers') # 'Sex_binarize',

    return df_merged


def add_race_ethnicity(dataframe):
    """add race and ethnicity to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe): must contain col `Identifiers`.
    Returns:
        dataframe (pd dataframe): returns `dataframe` with additional demographic columns
    """
    def race(x):
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
        
    def ethnicity(x):
        ethnicity_dict = {
            0: "White/Caucasian",
            1: "Hispanic or Latino",
            2: "Unknown",
            3: "Unknown",
            }
        return ethnicity_dict[x]

    # READ DEMOGRAPHICS - INTAKE INTERVIEW
    df_demo = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Parent_Measures/Interview_of_Emotional_and_Psychological_Function/PreInt_Demos_Fam.csv'))
    df_merged = dataframe.merge(df_demo, on='Identifiers', how='left')
    df_merged['PreInt_Demos_Fam,Child_Race_cat'] = df_merged['PreInt_Demos_Fam,Child_Race'].fillna(10).apply(lambda x: race(x)) # fill NaN values with "Unknown"
    df_merged['PreInt_Demos_Fam,Child_Ethnicity_cat'] = df_merged['PreInt_Demos_Fam,Child_Ethnicity'].fillna(3).apply(lambda x: ethnicity(x)) # fill NaN values with "Unknown"

    return df_merged


    df_score = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'CGAS.csv'))
    df_score.columns = df_score.columns.str.replace('CGAS,','')
    df_score[df_score['CGAS_Score']>100] = np.float("NaN")
    df_merged = df_score[['Identifiers', 'CGAS_Score']].merge(dataframe, on='Identifiers')

    return df_merged

