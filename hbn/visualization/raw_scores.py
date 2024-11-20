import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hbn.visualization import utils
from hbn.constants import Defaults 

def get_data(assessment='Parent', 
             dx='DX_ADHD',
             measure='CBCL',
             cols_to_filter=None,
             ):
    """get data from questionnaires
    Args:
        measure: str
        dx: str
        cols_to_filter: list
    returns:
        data (pd.DataFrame)
    """
    # get features and merge with demographics
    df = pd.read_csv(os.path.join(Defaults.INTERIM_FEATURES_DIR, 'participant_train_test.csv'))

    # # get features
    df_features = pd.read_csv(os.path.join(Defaults.INTERIM_FEATURES_DIR, f'{assessment}-features-raw.csv'), engine='python')

    # # get `cols_to_filter`
    if cols_to_filter is not None:
        cols_to_keep = [col for col in df_features.columns if any(string in col for string in cols_to_filter)]
        cols_to_keep.extend(['Identifiers'])
    elif (measure is not None) and (cols_to_filter is None):
        cols_to_keep = [col for col in df_features.columns if measure in col]
        cols_to_keep.extend(['Identifiers'])
    else:
        cols_to_keep = ['Identifiers']

    # merge with dataframe
    df_out = df.merge(df_features[cols_to_keep], on='Identifiers')

    # rename some columns
    df_out = df_out.rename(columns={'PreInt_Demos_Fam,Child_Race_cat': 'Race'})

    # remap 
    remap_labels = utils.remap_dx()
    def map_labels(x):
        if x in remap_labels.keys():
            return remap_labels[x]
        else:
            return x
    df_out[dx] = df_out[dx].apply(lambda x: map_labels(x))

    # reformat age
    df_out['Age_round'] = df_out['Age_round'].astype(int)

    return df_out, cols_to_keep

def map_labels(x):
    data_dict = {
    'Internalizing': 'Int',
    'Externalizing': 'Ext',
    'Withdrawn/Depressed': 'WD',
    'Social Problems': 'SP',
    'Anxious/Depressed': 'AD',
    'Thought Problems': 'TP',
    'Attention Problems': 'AP',
    'Rule Breaking': 'RBB',
    }
    return data_dict[x]

def melt_dataframe(df, dx='DX_Reading',data='Internalizing', key='Int'):

    # get all scores across parent, child, teacher
    all_cols = [f'CBCL,CBCL_{key}', f'YSR,YSR_{key}', f'TRF,TRF_{key}']

    # standarize (min, max scale)
    for col in all_cols:
        min_score = df[col].min()
        max_score = df[col].max()
        df[f'{col}_standarized'] = (df[col] - min_score) / (max_score - min_score)

    # melt parent, child, and teacher scores into one
    df_melt = pd.melt(df, value_vars=all_cols, id_vars=['Identifiers', dx, 'Race', 'Sex', 'Age_round']).rename(columns={'variable': data, 'value': f'{data}_standarized'})
    for k,v in {'YSR': 'Child', 'CBCL': 'Parent', 'TRF': 'Teacher'}.items():
        df_melt.loc[df_melt[data].str.contains(k), data] = v

    return df_melt

def load_parent_child_teacher(key=['Int'], dx='DX_Reading', dx_to_include=['Reading (all)', 'Reading (none)']):
    
    # get parent cols
    parent_cols = []
    for k in key:
        parent_cols.append(f'CBCL,CBCL_{k}')
    
    # get child cols
    child_cols = []
    for k in key:
        child_cols.append(f'YSR,YSR_{k}')

    # get teacher cols
    teacher_cols = []
    for k in key:
        teacher_cols.append(f'TRF,TRF_{k}')

    # get data
    df_parent = get_data(assessment='Parent', dx=dx, cols_to_filter=parent_cols)
    df_child = get_data(assessment='Child', dx=dx, cols_to_filter=child_cols)
    df_teacher = get_data(assessment='Teacher', dx=dx, cols_to_filter=teacher_cols)
    
    # merge dataframes
    df_merged = df_parent.merge(df_child).merge(df_teacher)

    # filter dataframe
    df_grouped = df_merged.groupby(['Identifiers', dx]).first().reset_index()
    df_grouped = df_grouped[(df_grouped[dx].isin(dx_to_include)) & 
                            (df_grouped['Age_round'].isin(np.arange(6,18)))]

    return df_grouped

def calculate_sex_percentage(df, groupby_cols=['Age_round', 'DX_Reading'], total_count='Sex'):
    """Calculates the percentage of females for each combination of age and diagnosis.

    Args:
        df: The pandas DataFrame containing the data.

    Returns:
        A pandas DataFrame with additional column 'female_percentage'.
    """

    # Create a groupby object based on 'age' and 'diagnosis'
    grouped_df = df.groupby(groupby_cols)

    # Calculate the number of females in each group
    female_counts = grouped_df['Sex'].apply(lambda x: (x == 'female').sum())
    male_counts = grouped_df['Sex'].apply(lambda x: (x == 'male').sum())

    # Calculate the female percentage within each DX
    if total_count=='Sex':
        female_percentage = (female_counts / grouped_df['Sex'].count()) * 100
        male_percentage = (male_counts / grouped_df['Sex'].count()) * 100
    elif total_count=='Age':
        total_counts_female = df.groupby('Age_round').apply(lambda x: (x['Sex'] == 'female').sum())
        total_counts_male = df.groupby('Age_round').apply(lambda x: (x['Sex'] == 'male').sum())
        female_percentage = (female_counts / total_counts_female) * 100
        male_percentage = (male_counts / total_counts_male) * 100


    # Add the female percentage as a new column to the original DataFrame
    df['female'] = df.set_index(groupby_cols).index.map(female_percentage)
    df['male'] = df.set_index(groupby_cols).index.map(male_percentage)

    # melt column into one
    melted_df = df.melt(id_vars=['Age_round', 'DX_ADHD', 'DX_Subtype_Name', 'DX_Cat_Name', 'split'],
                        value_vars=['female', 'male'], 
                        var_name='Sex', 
                        value_name='percent')

    return melted_df
