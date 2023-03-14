import os
import numpy as np
import pandas as pd
import logging

from hbn.constants import Defaults


def make_summary(save=True):
    """
    Save summary of dataset `(Diagnosis_ClinicianConsensus` + `Basic_Demos`) and save out participant identifiers
    Returns: 
        dx (pd dataframe)
    """
    
    # READ CLINICAL CONSENSUS
    dx_file = os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Diagnosis_ClinicianConsensus.csv')
    dx = pd.read_csv(dx_file)

    # do some clean up
    dx.columns = dx.columns.str.replace('Diagnosis_ClinicianConsensus,', '')

    # Replace "NaN" diagnosis with 'No Diagnosis Given: No Reason Given'
    dx.loc[dx['DX_01']==' ','DX_01'] = 'No Diagnosis Given: No Reason Given'
    dx.loc[dx['DX_01_Cat'].isna(),'DX_01_Cat'] = 'No Diagnosis Given: No Reason Given'

    # new disorder category
    diagnoses = [f'DX_{f:02}' for f in np.arange(1,11)]
    dx['comorbidities'] = dx[diagnoses].count(axis=1)-1

    # add demographics
    dx = _add_demographics(dataframe=dx)

    # bucket ages: over and under 10 yrs of age
    dx.loc[dx['Age']>10, 'Age_bracket'] = "over10"
    dx.loc[dx['Age']<=10, 'Age_bracket'] = "under10"

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


def make_demographics():
    """Get fullpath to clinical diagnosis and demographics and modify to save out specific columns (as numeric values)
    Returns:
        filename (str): includes cols ['Age', 'Sex', 'Race', 'Ethnicity', 'Diagnosis'], also saves file 'Demographic_Features.csv' in `out_dir`
    """
    # read in clinical diagnosis and demographics
    df = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Clinical_Diagnosis_Demographics.csv'))

    col_dict = {'Sex': 'Sex', 
                'Age': 'Age', 
                'DX_01': 'Diagnosis', 
                'comorbidities': 'comorbidities', 
                'PreInt_Demos_Fam,Child_Race_cat': 'Race', 
                'PreInt_Demos_Fam,Child_Ethnicity_cat': 'Ethnicity'
                } 
    for k,v in col_dict.items():
        df.loc[:,v] = df[k]
    df = pd.concat([df[['Identifiers']], df[col_dict.values()]], axis=1)

    # save to file
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv(os.path.join(Defaults.FEATURE_DIR, 'Demographic_Features.csv'), index=False)

    return 'Demographic_Features.csv'


def make_train_test_splits(out_dir=Defaults.MODEL_SPEC_DIR):
    """get train/validate and test identifiers (from dataframe output by `make_summary`), save them to file

    We try to balance the train/test splits 

    Args:
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

    # get dataframe containing all participants + diagnoses
    dataframe = make_summary(save=False)
    dataframe['Sex_binarize'] = dataframe['Sex'].map({'male': 0, 'female': 1})
    dataframe['DX_01_Cat_new_factorize'] = dataframe['DX_01_Cat_new'].factorize()[0]

    # get train/test for all participants
    strat_array = np.array(dataframe[['DX_01_Cat_new_factorize']])
    train_participants, test_participants, _, _ = train_test_split(dataframe['Identifiers'], dataframe['Identifiers'], test_size=0.2, random_state=42, stratify=strat_array)

    train_all_df = dataframe[dataframe['Identifiers'].isin(train_participants)].reset_index(drop=True)
    test_all_df = dataframe[dataframe['Identifiers'].isin(test_participants)].reset_index(drop=True)
    train_all_df['Identifiers'].to_csv(os.path.join(out_dir, 'train', f'train_participants-all.csv'), index=False)
    test_all_df['Identifiers'].to_csv(os.path.join(out_dir, 'test', f'test_participants-all.csv'), index=False)

    # get train/test separately for each disorder
    cols_to_group = ['DX_01_Cat_new', 'DX_01']
    for col in cols_to_group:
        for name, group in dataframe.groupby(col):

            # get diagnosis name
            outname = '_'.join(re.split(r'_|,|/| ', name))

            try: 
                # split train/test participants
                labels = np.array(group['Sex_binarize'])
                X_train, X_test, _, _ = train_test_split(group['Identifiers'], group['Identifiers'], test_size=0.2, random_state=42, stratify=labels)
                
                # get train and test dataframes
                X_train = group.merge(pd.DataFrame(X_train).reset_index(drop=True), on='Identifiers')
                X_test = group.merge(pd.DataFrame(X_test).reset_index(drop=True), on='Identifiers')

                # save to file
                X_train['Identifiers'].reset_index(drop=True).to_csv(os.path.join(out_dir, 'train', f'train_participants-{outname}.csv'), index=False)
                X_test['Identifiers'].reset_index(drop=True).to_csv(os.path.join(out_dir, 'test', f'test_participants-{outname}.csv'), index=False)
                print(f'writing train and test participants to file for {outname}')
            except:
                print(f'could not write out train and test participants for {outname} -- likely too few samples')


def get_disorder_categories():
    # get dataframe containing clinical diagnoses
    dataframe = make_summary()
    
    # get categories of diagnoses
    column='DX_01_Cat_new'
    categories = dataframe[column].unique().tolist()
    
    return categories + ['all'] 


def get_disorder(column='DX_01', category='Anxiety Disorders'):
    # get dataframe containing clinical diagnoses
    dataframe = make_summary()
    
    if category != 'all':
        disorders = dataframe[dataframe['DX_01_Cat_new']==category][column].unique()
    elif category=='all':
        disorders = dataframe[column].unique()
    
    return disorders


def get_participants(
    split='train', 
    disorders=['ADHD-Combined_Type', 'ADHD-Inattentive_Type'], 
    age='all',
    sex='all',
    path=Defaults.MODEL_SPEC_DIR
    ):
    """return list of participant identifiers and filter based on `disorders`, `age`, `sex`

    Args:  
        split (str): default is 'train', other option is 'test' or 'all'
        disorders (list of str): list of diagnoses
        age (int or 'all'): (optional): default is 'all'. other options are list of numbers between 6 - 21
        sex (str or 'all'): (optional): default is 'all'. other options 'male' or 'female
    Returns:
        `identifiers` (list of str): participant list
    """
    import os
    import pandas as pd

    if split=='all':
        split = ['train', 'test']
    elif not isinstance(split, list):
        split = [split]

    df_all = pd.DataFrame()
    # loop over disorders
    for disorder in disorders:
        for sp in split:
            #name = '_'.join(re.split(r'_|,|/| ', disorder))
            fname = os.path.join(path, sp, f'{sp}_participants-{disorder}.csv')
            if os.path.isfile(fname):
                df = pd.read_csv(fname)

                # load clinical diagnosis
                dx = make_summary(save=False)
            
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


def define_new_categories(dataframe):
    """define new disorder categories using labels from `DX_01_Cat`

    Args:
        dataframe (pd dataframe)
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

    ## divide neurodevelopmental disorders into other categories
    dataframe['DX_01_Cat_new'] = dataframe.apply(lambda x: new_categories(x['DX_01'], x['DX_01_Cat']), axis=1)

    dx_to_model = ['Anxiety Disorders', 'Autism Spectrum Disorder', 'ADHD', 'No Diagnosis Given: No Reason Given',
                                        'No Diagnosis Given', 'No Diagnosis Given: Incomplete Eval',
                                        'Specific Learning Disorder with Impairment in Reading']
    dx_not_to_model = dataframe[~dataframe['DX_01_Cat_new'].isin(dx_to_model)].reset_index(drop=True)
    dx_not_to_model['dx_model'] = False

    dx_model = dataframe[dataframe['DX_01_Cat_new'].isin(dx_to_model)].reset_index(drop=True)
    dx_model['dx_model'] = True

    df_concat = pd.concat([dx_model, dx_not_to_model])

    return df_concat


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
    df_demo = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Parent_Measures/Demographic_Questionnaire_Measures/Basic_Demos.csv'))
    df_demo.columns = df_demo.columns.str.replace('Basic_Demos,','')
    df_demo['Sex'] = df_demo['Sex'].map({0: 'male', 1: 'female'})
    df_merged = df_demo[['Identifiers', 'Age', 'Sex', 'Enroll_Year']].merge(dataframe, on='Identifiers') # 'Sex_binarize',

    return df_merged


def _add_race_ethnicity(dataframe):
    """add race and ethnicity to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe): must contain col `Identifiers`
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


def add_CGAS_Score(dataframe):
    """add CGAS_Score to existing dataframe, merging on participant id `Identifiers`

    Args: 
        dataframe (pd dataframe): must contain col `Identifiers`
    Returns: 
        returns `dataframe` with additional `CGAS_Score` column
    """

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
