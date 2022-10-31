import os
from this import d
import numpy as np
import pandas as pd

from hbn.constants import Defaults


def make_summary():
    """
    Save summary of dataset (clinical diagnosis + demographics) and save out participant identifiers: `Clinical_Diagnosis.csv` is parsed from master data (`phenotype.parse_data`) but is incorrect. `Clinical_Diagnosis_2022.csv`
    was downloaded directly from LORIS and is correct, the latter is returned by this function.
    Returns: 
        dx (pd dataframe)
    """
    
    # READ CLINICAL CONSENSUS
    dx_file = os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Clinical_Diagnosis_2022.csv')
    dx = pd.read_csv(dx_file)

    # do some clean up
    # dx.columns = dx.columns.str.replace('ConsensusDx,', '')
    dx.columns = dx.columns.str.replace('Diagnosis_ClinicianConsensus,', '')
    dx['Identifiers'] = dx['Identifiers'].str.strip(',assessment')

    # new disorder category
    diagnoses = [f'DX_{f:02}' for f in np.arange(1,11)]
    dx['comorbidities'] = dx[diagnoses].count(axis=1)-1

    # optionally add demographics
    dx = _add_demographics(dataframe=dx)

    # deal with missing values and NaN
    dx = dx.replace(' ', np.float("NaN")).fillna(np.float("NaN")).dropna(how='all', axis=1)
    dx = dx.dropna(how='all', axis=0)
    
    # save out new files to disk
    # participants
    dx['Identifiers'].to_csv(os.path.join(Defaults.PHENO_DIR, 'participants.csv'))
    # updated clinical diagnosis
    dx.to_csv(os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Clinical_Diagnosis_Demographics.csv'))

    return dx


def parse_phenotypic_data(
    assessments=['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'], 
    ):
    """parse phenotype assessments

    Args: 
        assessments (list of str): options: ['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures']
    Returns: 
        dataframe (pd dataframe): `assessment` parsed and saved to disk
    """
    # loop over assessments
    for assessment in assessments:
        Abbreviation = 'Abbreviation(s) COINS'
        if assessment=="Teacher Measures":
            Abbreviation='Abbreviation'

        # set up directory
        assessment_dir = os.path.join(Defaults.PHENO_DIR, '_'.join(assessment.split()))
        if not os.path.isdir(assessment_dir):
            os.makedirs(assessment_dir)

        # load in master dataframe
        df = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'data-2022-08-24T16_37_18.263Z.csv'));
        df = df.replace('.', np.float("NaN")) # replace '.' with NaN (easier to drop these rows)
        df['Identifiers'] = df['Identifiers'].str.strip(r',assessment|,,assessment|')

        # load excel containing descriptions of phenotypic assessment
        info, domain = assessment_list(assessment=assessment)

        # parse columns
        info['abbrev_parsed'] = info[Abbreviation].str.split(r'_|,')
        info['measure_parsed'] = info['Measure'].str.split(r'_|,|/| ')
        if domain:
            info['domain_parsed'] = info['Domain'].str.split(r'_|,|/| ')
        
        # loop over rows
        for row in info.index:
            abbrev = info.iloc[row]['abbrev_parsed'][0] 
            
            # create separately outdir if `Domain` is present
            out_dir = assessment_dir
            if domain:
                domain_name = info.iloc[row]['domain_parsed']
                while("" in domain_name) :
                    domain_name.remove("") 
                out_dir = os.path.join(assessment_dir, '_'.join(domain_name))
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)

            # subset the dataframe based on `Abbreviation`
            df_subset = df.filter(like=abbrev)
            df_subset = pd.concat([df['Identifiers'], df_subset], axis=1).set_index('Identifiers') # add identifiers

            # only save out datasets that aren't empty
            if not df_subset.empty:
                df_subset = df_subset.dropna(how='all').reset_index() # drop rows where all values are missing
                df_subset.to_csv(os.path.join(out_dir, '_'.join(info.iloc[row]['measure_parsed'])) + '.csv', index=None)
                print(f'saving to dir {out_dir}')


def assessment_list(assessment, save=True):
    """correct assessment list, update `domain` for each `measure`

    Args:
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'
        save (bool): default is True. save to disk in `data/phenotype`
    Returns: 
        info (pd dataframe), domain (str)
    """
    fname = 'Assessment_List_Jan2019'
    # parse `assessment` sheets
    xls = pd.ExcelFile(os.path.join(Defaults.PHENO_DIR, f'{fname}.xlsx'), engine='openpyxl');
    
    # read excel
    info = pd.read_excel(xls, assessment, header=1);
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
    
    if save:
        assessment = '_'.join(assessment.split())
        info.to_csv(os.path.join(Defaults.PHENO_DIR, f'{fname}_{assessment}.csv'))

    return info, domain


def _add_demographics(dataframe):
    """add demographics to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe): must contain col `Identifiers`
    Returns:
        dataframe (pd dataframe): returns `dataframe` with additional demographic columns
    """
    # READ BASIC DEMOGRAPHICS
    df_demo = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Parent_Measures/Demographic_Questionnaire_Measures/Demographics.csv'))
    df_demo.columns = df_demo.columns.str.replace('Basic_Demos,','')
    df_demo['Sex'] = df_demo['Sex'].map({0: 'male', 1: 'female'})
    df_merged = df_demo[['Identifiers', 'Age', 'Sex', 'Enroll_Year']].merge(dataframe, on='Identifiers') # 'Sex_binarize',

    return df_merged


def add_CGAS_Score(dataframe):
    """add CGAS_Score to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe): must contain col `Identifiers`
    Returns: 
        returns `dataframe` with additional `CGAS_Score` column
    """

    df_score = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures/Children\'s_Global_Assessment_Scale.csv'))
    df_score.columns = df_score.columns.str.replace('CGAS,','')
    df_score[df_score['CGAS_Score']>100] = np.float("NaN")
    df_merged = df_score[['Identifiers', 'CGAS_Score']].merge(dataframe, on='Identifiers')

    return df_merged


def _dx_grouping(x):
    """group diagnoses into broader set of domains

    Args: 
        x (str): diagnosis name. e.g., 'Social Phobia'
    Returns: 
        k (str): one of keys from `remap_dict`
    """
    remap_dict = {
                'adhd': ['ADHD', 'Attention-Deficit'],
                'language_communication': ['Tourettes', 'Speech', 'Communication', 'Mutism', ' Tic Disorder', 'Language'],
                'anxiety': ['Stress', 'Adjustment', 'Agoraphobia', 'Obsessive', 'Panic', 'Anxiety', 'Specific Phobia'],
                'asd': ['Autism'],
                'mood': ['Bipolar I', 'Bipolar II', 'Cyclothymic', 'Depressive', 'Mood'],
                'no_diagnosis': ['No Diagnosis'],
                'conduct_relational': ['Conduct', 'Intermittent Explosive', 'Oppositional Defiant', 'Relational', 'Attachment'],
                'body_related': ['Encopresis', 'Enuresis', 'Excoriation', 'Food Intake', 'Bulimia', 'Dysphoria'],
                'substance_use': ['Alcohol', 'Cannabis', 'Substance'],
                'intellectual': ['Intellectual', 'Learning', 'Neurocognitive', 'Neurodevelopmental'],
                'psychosis': ['Delirium', 'Schizophrenia']
                }
    
    for k,v in remap_dict.items():
        for vv in v:
            if vv in x:
                return k