import os
import numpy as np
import pandas as pd
import logging

from hbn.constants import Defaults


def make_summary(fpath=None, save=True):
    """
    Save summary of dataset `(Diagnosis_ClinicianConsensus` + `Basic_Demos`) and save out participant identifiers
    Returns: 
        fpath (str or None): fullpath to `Diagnosis_ClinicianConsensus.csv` file. If None, then looks in Default directory.
        dx (pd dataframe)
    """
    from pathlib import Path
    
    # READ CLINICAL CONSENSUS
    if fpath is None:
        fpath = os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Diagnosis_ClinicianConsensus.csv')
    dx = pd.read_csv(fpath)

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
    dx = _add_demographics(dataframe=dx)

    # bucket ages: early, emerging, and fluent readers
    dx.loc[dx['Age']>=10, 'Age_bracket'] = "over10"
    dx.loc[dx['Age']<10, 'Age_bracket'] = "under10"

    # deal with missing values and NaN
    dx = dx.replace(' ', np.float("NaN")).fillna(np.float("NaN")).dropna(how='all', axis=1)
    dx = dx.dropna(how='all', axis=0)

    # add new categories (including categories to be modeled)
    dx = define_new_categories(dataframe=dx)

    # add ethnicity
    dx = _add_race_ethnicity(dataframe=dx)

    # participants
    dx = dx.loc[:, ~dx.columns.str.contains('^Unnamed')]
    
    # save out new files to disk
    if save:
        dx['Identifiers'].to_csv(os.path.join(Defaults.PHENO_DIR, 'participants.csv'))
        # updated clinical diagnosis
        dx.to_csv(os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Clinical_Diagnosis_Demographics.csv'), index=False)

    return dx


def get_all_diagnoses(dataframe=None):
    """Get all diagnoses from dataframe

    Args:
        dataframe (pd dataframe): must contain cols `DX_{num_str}_Cat_new` and `DX_{num_str}`. If None, will use `make_summary`
    Returns: 
        diagnoses_all (dict): dict of all diagnoses and subtypes
    """
    import numpy as np

    # get dataframe if None is given
    if dataframe is None:
        dataframe = make_summary(save=False)

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


def filter_dataframe(dataframe=None, column='diagnosis', value='ADHD'):
    """ Return participant identifiers for either `diagnosis` (e.g., ADHD) or `subtype` ('ADHD-Combined Type')

    Args:
        dataframe (pd dataframe or None): must contain columns `DX_{num_str}_Cat_new` and `DX_{num_str}`. If None, will use `make_summary`
        column (str): must be either 'diagnosis' or 'subtype'
        value (str): e.g. 'ADHD'    
    Returns:
        dataframe_filtered (pd dataframe): contains columns `Identifiers`
    """

    # get dataframe if None is given
    if dataframe is None:
        dataframe = make_summary(save=False)

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


def make_train_test_splits(dataframe=None, diagnoses=None, out_dir=Defaults.MODEL_SPEC_DIR):
    """get train/validate and test identifiers (from dataframe output by `make_summary`), save them to file

    We try to balance the train/test splits 

    Args:
        dataframe (pd DataFrame or None): dataframe containing all participants + diagnoses. if None, will use `make_summary` function
        diagnoses (list of str or None): list of diagnoses. If None, will use `get_all_diagnoses`
        out_dir (str): full path to out dir where train and test identifiers will be stored. default is `MODEL_SPEC_DIR`
    """
    import re
    import os
    import pandas as pd
    from hbn import io
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedShuffleSplit

    io.make_dirs(os.path.join(out_dir, 'train'))
    io.make_dirs(os.path.join(out_dir, 'test'))

    if dataframe is None:
        # get dataframe containing all participants + diagnoses
        dataframe = make_summary(save=False)

    # get all diagnoses and subtypes
    if diagnoses is None:
        diagnoses = get_all_diagnoses(dataframe)

    dataframe['Sex_binarize'] = dataframe['Sex'].map({'male': 0, 'female': 1})

    # loop over diagnoses and subtypes
    for key in diagnoses.keys():
        for diag in diagnoses[key]:

            # get diagnosis name
            outname = '_'.join(re.split(r'_|,|/| ', diag))

            try: 
                # get participant identifiers
                group = filter_dataframe(dataframe, column=key, value=diag)

                # get labels to stratify
                labels = np.array(group['Sex_binarize'])

                # split train/test participants
                X_train, X_test, _, _ = train_test_split(group['Identifiers'], group['Identifiers'], test_size=0.2, random_state=42, stratify=labels)

                # save to file
                X_train.reset_index(drop=True).to_csv(os.path.join(out_dir, 'train', f'train_participants-{outname}.csv'), index=False)
                X_test.reset_index(drop=True).to_csv(os.path.join(out_dir, 'test', f'test_participants-{outname}.csv'), index=False)
                print(f'writing train and test participants to file for {outname}')
            except:
                print(f'could not write out train and test participants for {outname} -- likely too few samples')


def make_items(fpath=None, out_dir=Defaults.SUBTYPE_DIR):
    """make item fname (modified from original https://github.com/charlie42/diagnosis-predictor/blob/main/references/item-names.csv)
    """
    # load item names fname
    if fpath is None:
        fpath = os.path.join(Defaults.PHENO_DIR, 'item-names.csv')
    dataframe = pd.read_csv(fpath)

    # add new assessment, domain, measures info to item names
    df = _match_datadic_to_data(dataframe=df)

    # add proprietry/free questionnaires to item names
    outpath = os.path.join(Defaults.SUBTYPE_DIR, 'Free_Assessments_HBN_new.csv')
    fpath = os.path.join(Defaults.PHENO_DIR, 'Free_Assessments_HBN.xlsx')
    if not os.path.isfile(outpath):
        make_new_proprietary_assessments_file(fpath, outpath)
    df_proprietary = pd.read_csv(outpath)

    # add proprietary info to dataframe and assign NaN to Unknown
    df = df_proprietary.merge(df, on='datadic', how='outer')

    # remove rows that don't have any questions or keys
    conditional = (df['questions'].isna()) & (df['keys'].isna())
    df = df[~conditional]

    # fix domain name (missing domain in `Assessment_List_Jan2019.xlsx`)
    df = _fix_domain(dataframe=df)

    # identify questions that contain total scores
    df = _identify_total_scores(dataframe=df)

    # identify questions that are subheadings
    df = _identify_subheadings(dataframe=df)

    # identify questions that are preambles
    df = _identify_preamble(dataframe=df)

    # save out new file
    df.to_csv(os.path.join(out_dir, 'item-names-cleaned.csv'), index=False)


def make_interim_data_files():
    import os
    from hbn.features import feature_selection
    from hbn.constants import Defaults
    from hbn import io

    ## save out data files (link with dictionary keys) for different feature specs
    assessments = ['Parent', 'Child', 'Teacher']
    data_dict = {True: 'preprocessed', False: 'raw'}

    for assessment in assessments:
        for k,v in data_dict.items():
        
            feature_spec = os.path.join(Defaults.FEATURE_DIR, f'features-{assessment}_Measures-all-all-all-spec.json')

            # preprocessed and raw data
            feature_info = io.read_json(feature_spec)
            feature_info['preprocessing']['preprocess'] = k
            df = feature_selection.phenotype_features(feature_spec=feature_info, drop_identifiers=False)
            df.columns = [col.replace("numeric__", "") for col in df.columns]
            
            df.to_csv(os.path.join(Defaults.SUBTYPE_DIR, f'{assessment}-features-{v}.csv'), index=False)
            print(f'saving out {assessment}-features-{v}.csv to disk')


def get_participants(
    dataframe=None,
    split='train', 
    disorders=['ADHD-Combined_Type', 'ADHD-Inattentive_Type'], 
    age='all',
    sex='all',
    path=Defaults.MODEL_SPEC_DIR
    ):
    """return list of participant identifiers and filter based on `disorders`, `age`, `sex`

    Args:  
        dataframe (pd dataframe or None): default is None. Must contain `Age`, `Sex`, and `Identifiers`. If None, will use `make_summary`
        split (str): default is 'train', other option is 'test' or 'all'
        disorders (list of str): list of diagnoses. get options from `get_all_diagnoses`
        age (int or 'all'): (optional): default is 'all'. other options are list of numbers between 6 - 21
        sex (str or 'all'): (optional): default is 'all'. other options 'male' or 'female
    Returns:
        `identifiers` (list of str): participant list
    """
    import os
    import re
    import pandas as pd

    if split=='all':
        split = ['train', 'test']
    elif not isinstance(split, list):
        split = [split]

    # if 'All_Other_Diagnoses' is given, then return all participants
    if 'All_Other_Diagnoses' in disorders:
        disorders = ['all']

    # check if `disorders` is coded correctly
    disorders = ['_'.join(re.split(r'_|,|/| ', d)) for d in disorders]
        
    df_all = pd.DataFrame()
    # loop over disorders
    for disorder in disorders:
        for sp in split:
            fname = os.path.join(path, sp, f'{sp}_participants-{disorder}.csv')
            if os.path.isfile(fname):
                df = pd.read_csv(fname)

                # load clinical diagnosis
                if dataframe is None:
                    dx = make_summary(save=False)
                else:
                    dx = dataframe
            
                # integrate dataframes
                df_dx = df.merge(dx, on=['Identifiers'])
                
                # filter on age
                if age != 'all':
                    if not isinstance(age, list):
                        age = [age]
                    df_dx['Age'] = df_dx['Age'].round()
                    df_dx = df_dx[df_dx['Age'].isin(age)]

                # filter on sex
                if sex != 'all':
                    df_dx = df_dx[df_dx['Sex']==sex]
            
                df_all = pd.concat([df_dx, df_all])
            else:
                print(f'{fname} does not exist')
    
    identifiers = df_all.reset_index(drop=True)['Identifiers'].tolist()

    return identifiers


def add_participant_groups(
    participants,
    dataframe=None,
    disorders=['ADHD', 'All_Other_Diagnoses'], 
    ):
    """ add participant groups to participant identifiers if one of the disorders is 'All_Other_Diagnoses'
    Args:
        participants (list of str): list of participant identifiers (output from `get_participants`)
        dataframe (pd dataframe or None): default is None. Must contain `DX_{num_str}_Cat_new` and `DX_{num_str}` and `Identifiers`. If None, will use `make_summary`
        disorders (list of str): list of disorders, must include 'All_Other_Diagnoses'
    Returns:
        df (pd dataframe): contains columns: 'Identifiers' and 'participant_groups'
    """

    # make pandas dataframe
    df = pd.DataFrame(participants, columns=['Identifiers'])

    # load clinical diagnosis
    if dataframe is None:
        dx = make_summary(save=False)
    else:
        dx = dataframe

    # merge on clinical diagnosis
    df = dx.merge(df, on=['Identifiers'])

    # get all disorders other than 'All_Other_Diagnoses
    disorders_to_leave_out = [d for d in disorders if d!='All_Other_Diagnoses']

    # which column are we using?
    columns = ['DX_01', 'DX_01_Cat_new']
    for col in columns:
        disorders_present = sum(df[col].isin(disorders_to_leave_out))
        if disorders_present>1:
            break

    # index disorders
    for disorder in disorders_to_leave_out:

        # disorder to leave out should match unique values in df[col]
        disorder = disorder.replace('_', ' ')

        idx = df[col].isin([disorder])
        df.loc[idx, 'participant_groups'] = disorder
        df.loc[~idx, 'participant_groups'] = 'All_Other_Diagnoses'

    return df[['Identifiers', 'participant_groups']]


def define_new_categories(dataframe=None):
    """define new disorder categories using labels from `DX_<num>_Cat`

    Args:
        dataframe (pd dataframe or None): default is None. Must contain columns `DX_{num_str}_Cat_new` and `DX_{num_str}`. If None, will use `make_summary`
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

    # get dataframe if None is given
    if dataframe is None:
        dataframe = make_summary(save=False)

    for num in range(1,11):
        
        num_str = str(num).zfill(2)
        
        ## divide neurodevelopmental disorders into other categories
        dataframe[f'DX_{num_str}_Cat_new'] = dataframe.apply(lambda x: new_categories(x[f'DX_{num_str}'], x[f'DX_{num_str}_Cat']), axis=1)
        
        # removed column 'to_model' from dataframe

    return dataframe


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


def _match_datadic_to_data(dataframe):
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
    
    # get dataframe if None is given
    if dataframe is None:
        dataframe = make_summary(save=False)

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


def _fix_domain(dataframe):
    measures_to_change = ['SympChck', 'ICU_P', 'ARI_P', 'SRS_Pre', 'SRS', 'RBS', 'SDQ', 'WHODAS_P', 'SAS', 
                'CIS_P', 'SCQ', 'ASSQ', 'SWAN','ESWAN','SCARED_P','MFQ_P', 'CBCL', 'CBCL_Pre']
    for abbrev in dataframe['datadic'].unique():
        if abbrev in measures_to_change:
            dataframe.loc[(dataframe["datadic"]==abbrev) & (~dataframe['keys'].isna()), "domains"] = 'Questionnaire_Measures_of_Emotional_and_Cognitive_Status'

    return dataframe


def _identify_total_scores(dataframe):

    overall_scores = ['Total', 'Raw Score', 'T-Score', 'T Score', 'Standard']

    # filter dataframe
    questions_to_filter = dataframe[dataframe['questions'].str.contains('|'.join(overall_scores))==True]['questions'].unique()

    # assign new column to identify whether qustion is a total score or not
    dataframe['Total_Scores'] = dataframe['questions'].isin(questions_to_filter)
    dataframe.loc[dataframe['Total_Scores']==False, 'Not_Total_Scores'] = True

    return dataframe


def _identify_preamble(dataframe):

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


def _identify_subheadings(dataframe):

    # find keys with exact match: "Scores", "Scale Score", "Scoring"
    # and find keys that contain "Scales"
    conditional = (dataframe['questions']=='Scores') | (dataframe['questions'].str.contains('Scale Scores')) | (dataframe['questions']=='Scoring') |  (dataframe['questions'].str.contains('Scales')) & (dataframe['keys'].isna())
    
    # filter dataframe
    questions_to_filter = dataframe[conditional]['questions'].unique()

    # assign new column to identify whether qustion is a total score or not
    dataframe['Subheadings'] = dataframe['questions'].isin(questions_to_filter)

    return dataframe


def make_new_proprietary_assessments_file(fpath=None, outpath=None):
    """
    Args:
        fpath (str or None): fullpath to `Free_Assessments_HBN.xlsx` file. If None, looks in `Defaults.PHENO_DIR`
        outpath (str or None): save file to `outpath`. If None, saves to `Defaults.SUBTYPE_DIR` as `Free_Assessments_HBN_new.csv`
    """

    # get full path to proprietary data
    if fpath is None:
        fpath = os.path.join(Defaults.PHENO_DIR, 'Free_Assessments_HBN.xlsx')

    # get savedir
    if outpath is None:
        outpath = os.path.join(Defaults.SUBTYPE_DIR, 'Free_Assessments_HBN_new.csv')
    
    # read in data
    df = pd.read_excel(fpath)

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


def _add_demographics(dataframe=None):
    """add demographics to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe or None): default is None. Must contain columns `Identifiers`. If None, will use `make_summary`
    Returns:
        dataframe (pd dataframe): returns `dataframe` with additional demographic columns
    """
    # READ BASIC DEMOGRAPHICS
    df_demo = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Parent_Measures/Demographic_Questionnaire_Measures/Basic_Demos.csv'))
    df_demo.columns = df_demo.columns.str.replace('Basic_Demos,','')
    df_demo['Sex'] = df_demo['Sex'].map({0: 'male', 1: 'female'})

    # get dataframe if None is given
    if dataframe is None:
        dataframe = make_summary(save=False)

    df_merged = df_demo[['Identifiers', 'Age', 'Sex', 'Enroll_Year']].merge(dataframe, on='Identifiers') # 'Sex_binarize',

    return df_merged


def _add_race_ethnicity(dataframe=None):
    """add race and ethnicity to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe): must contain col `Identifiers`. If None, will use `make_summary`
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

    # get dataframe if None is given
    if dataframe is None:
        dataframe = make_summary(save=False)

    # READ DEMOGRAPHICS - INTAKE INTERVIEW
    df_demo = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Parent_Measures/Interview_of_Emotional_and_Psychological_Function/PreInt_Demos_Fam.csv'))
    df_merged = dataframe.merge(df_demo, on='Identifiers', how='left')
    df_merged['PreInt_Demos_Fam,Child_Race_cat'] = df_merged['PreInt_Demos_Fam,Child_Race'].fillna(10).apply(lambda x: race(x)) # fill NaN values with "Unknown"
    df_merged['PreInt_Demos_Fam,Child_Ethnicity_cat'] = df_merged['PreInt_Demos_Fam,Child_Ethnicity'].fillna(3).apply(lambda x: ethnicity(x)) # fill NaN values with "Unknown"

    return df_merged


def add_CGAS_Score(dataframe=None):
    """add CGAS_Score to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe): must contain col `Identifiers`
    Returns: 
        returns `dataframe` with additional `CGAS_Score` column
    """

    # get dataframe if None is given
    if dataframe is None:
        dataframe = make_summary(save=False)

    df_score = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'CGAS.csv'))
    df_score.columns = df_score.columns.str.replace('CGAS,','')
    df_score[df_score['CGAS_Score']>100] = np.float("NaN")
    df_merged = df_score[['Identifiers', 'CGAS_Score']].merge(dataframe, on='Identifiers')

    return df_merged


def _setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
