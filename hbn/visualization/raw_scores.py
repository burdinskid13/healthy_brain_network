import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hbn.visualization import utils
from hbn.constants import Defaults 

def get_data(release='Release11_Apr2024',
             assessment='Parent', 
             cols_to_filter=None,
             vals_to_filter=None,
             measure='CBCL',
             features=None,
             ):
    """get data from questionnaires specific to `release` and `assessment`. optionally filters dataframe using `cols_to_filter`
    which contains a list of columns to filter (e.g., ['age', 'sex']) and corresponding `vals_to_filter` which contains lists of 
    vals to filter from each column (e.g., [[6,7,8,9,10], ['male']]). if you want to filter the dataframe to include only certain `features`
    then assign `features` to a list of features to keep

    Args:
        release: str
        assessment: str
        cols_to_filter: dict or None
        measure: str
        features: list or None
    returns:
        data (pd.DataFrame)
    """
    from hbn.features.build_features import _index_dataframe_by_columns_values
    
    # get demographics
    df = pd.read_csv(os.path.join(Defaults.INTERIM_FEATURES_DIR, release, 'participant_train_test.csv'))

    # filter demographics
    if (cols_to_filter is not None) and (vals_to_filter is not None):
        df = _index_dataframe_by_columns_values(dataframe=df, columns=cols_to_filter, values_list=vals_to_filter)

    # # get features
    df_features = pd.read_csv(os.path.join(Defaults.INTERIM_FEATURES_DIR, release, f'{assessment}-features-raw.csv'), engine='python')

    # # get `cols_to_filter`
    if features is not None:
        cols_to_keep = [col for col in df_features.columns if any(string in col for string in features)]
        cols_to_keep.extend(['Identifiers'])
    elif (measure is not None) and (features is None):
        cols_to_keep = [col for col in df_features.columns if measure in col]
        cols_to_keep.extend(['Identifiers'])
    else:
        cols_to_keep = ['Identifiers']

    # merge with dataframe
    df_out = df.merge(df_features[cols_to_keep], on='Identifiers', how='inner')

    # # remap 
    # remap_labels = utils.remap_dx()
    # def map_labels(x):
    #     if x in remap_labels.keys():
    #         return remap_labels[x]
    #     else:
    #         return x
    # df_out[dx] = df_out[dx].apply(lambda x: map_labels(x))

    return df_out.reset_index(drop=True)

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


def melt_dataframe(df, dx='DX_Reading', data='Internalizing', key='Int'):

    # get all scores across parent, child, teacher
    all_cols = [f'CBCL,CBCL_{key}', f'YSR,YSR_{key}', f'TRF,TRF_{key}']

    # standarize (min, max scale)
    for col in all_cols:
        min_score = df[col].min()
        max_score = df[col].max()
        df[f'{col}_standarized'] = (df[col] - min_score) / (max_score - min_score)

    # melt parent, child, and teacher scores into one
    df_melt = pd.melt(df, value_vars=all_cols, id_vars=['Identifiers', dx, 'race', 'sex', 'age_round']).rename(columns={'variable': data, 'value': f'{data}_standarized'})
    for k,v in {'YSR': 'Child', 'CBCL': 'Parent', 'TRF': 'Teacher'}.items():
        df_melt.loc[df_melt[data].str.contains(k), data] = v

    return df_melt


def load_parent_child_teacher_CBCL(key=['Int'], dx='DX_Reading', dx_to_include=['Reading (all)', 'Reading (none)']):
    
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
    df_parent = get_data(assessment='parent', dx=dx, measure=None, cols_to_filter=parent_cols)
    df_child = get_data(assessment='child', dx=dx, measure=None, cols_to_filter=child_cols)
    df_teacher = get_data(assessment='teacher', dx=dx, measure=None, cols_to_filter=teacher_cols)
    
    # merge dataframes
    df_merged = df_parent.merge(df_child).merge(df_teacher)

    # filter dataframe
    df_grouped = df_merged.groupby(['Identifiers', dx]).first().reset_index()
    keyboard # FLAG USE OF .FIRST() HERE
    df_grouped = df_grouped[(df_grouped[dx].isin(dx_to_include)) & 
                            (df_grouped['age_round'].isin(np.arange(6,18)))]

    return df_grouped


def calculate_sex_percentage(df_compare, df_overall, groupby='age_round'):
    """Calculates the percentage of females for each combination of age and diagnosis.

    Args:
        df: The pandas DataFrame containing the data.
        groupby (list of str or str or None): 'age_round' etc.
    Returns:
        A pandas DataFrame with additional column 'female_percentage'.
    """
    # optionally groupby 'age' for example
    if groupby is None:
        female_counts = df_compare.groupby(groupby)['sex'].apply(lambda x: (x == 'female').sum())
        male_counts = df_compare.groupby(groupby)['sex'].apply(lambda x: (x == 'male').sum())
    
        female_counts_overall = df_overall.groupby(groupby)['sex'].apply(lambda x: (x == 'female').sum())
        male_counts_overall = df_overall.groupby(groupby)['sex'].apply(lambda x: (x == 'male').sum())

    # Calculate the number of females in each group
    female_counts = df_compare.groupby(groupby)['sex'].apply(lambda x: (x == 'female').sum())
    male_counts = df_compare.groupby(groupby)['sex'].apply(lambda x: (x == 'male').sum())
    
    female_counts_overall = df_overall.groupby(groupby)['sex'].apply(lambda x: (x == 'female').sum())
    male_counts_overall = df_overall.groupby(groupby)['sex'].apply(lambda x: (x == 'male').sum())
    
    female_percentage = pd.Series([0]); male_percentage = pd.Series([0])
    if not female_counts.empty: 
        female_percentage = female_counts / (female_counts_overall) *100
    if not male_counts.empty:
        male_percentage = male_counts / (male_counts_overall) *100

    # create new dataframe
    df_f = female_percentage.reset_index(name='perc') \
        .assign(sex='female', count=female_counts.reset_index(name='count')['count'], overall_count=female_counts_overall.reset_index(name='overall_count')['overall_count'])
    df_f['sex_plot'] = df_f['sex'].map({'female': f'female (n={str(sum(female_counts))})'})

    df_m = male_percentage.reset_index(name='perc') \
        .assign(sex='male', count=male_counts.reset_index(name='count')['count'], overall_count=male_counts_overall.reset_index(name='overall_count')['overall_count'])
    df_m['sex_plot'] = df_m['sex'].map({'male': f'male (n={str(sum(male_counts))})'})

    df_concat = pd.concat([df_f, df_m]).reset_index(drop=True).dropna()

    return df_concat
