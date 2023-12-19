
def stratify_split(
    df, 
    columns_to_stratify=[], 
    test_size=0.2, 
    random_state=42
    ):
    """ conventional stratified split using `from sklearn.model_selection import train_test_split`

    Args:
        df (pd dataframe): dataframe to stratify
        columns_to_stratify (list): A list of columns to use for stratified sampling
        test_size (float): The proportion of the data to be included in the test set. Defaults to 0.2.
        random_state (int): The random seed to use for splitting the data. Defaults to 42.
    Returns:
        df_out (pd dataframe): concatenated train and test dataframes. contains column `split` to indicate which rows are train and test
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    stratify = None
    if len(columns_to_stratify) > 0:
        stratify = df[columns_to_stratify]

    # Split the DataFrame into train and test sets using stratified sampling.
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)

    # Assign split and concat dataframes
    train_df['split'] = 'train'
    test_df['split'] = 'test'

    df_out = pd.concat([train_df, test_df]).reset_index(drop=True)

    return df_out


def get_unique_dataset(
    df,
    participant_col='Identifiers',
    col_to_group='DX_Cat_Name',
    cols_to_keep=['Sex', 'Age', 'PreInt_Demos_Fam,Child_Race_cat'],
    test_size=.2,
    random_state=42
    ):
    """ stratify split on unique participants

    Args:
        df (pd dataframe): dataframe to stratify
        test_size (float): The proportion of the data to be included in the test set. Defaults to 0.2.
        random_state (int): The random seed to use for splitting the data. Defaults to 42.
    """
    import itertools
    import pandas as pd

    # get cols to keep, `columns_to_stratify` and `participant_col`
    cols_to_keep_with_part = [[participant_col], cols_to_keep]
    cols_to_keep_with_part = list(itertools.chain(*cols_to_keep_with_part))
    
    df_clean = df.dropna()
    dx_index = df_clean.groupby(col_to_group)[participant_col].count().sort_values().reset_index().reset_index().set_index(col_to_group)['index']
    df_clean['DX_Index'] = df_clean[col_to_group].replace(dx_index)
    df_repdx = df_clean.groupby(participant_col)['DX_Index'].min().reset_index().drop_duplicates().set_index(participant_col)
    
    # get unique dataset
    df_unique = df[cols_to_keep_with_part].drop_duplicates().set_index(participant_col).join(df_repdx).reset_index(drop=False)

    return df_unique


def index_into_original_dataframe(df_original, df_stratify):
    """ Index train and test participants from `df_stratify` into `df_original`.

    Args:
        df_original (pd dataframe): original dataframe
        df_stratify (pd dataframe): stratified dataframe
    Returns:
        df_out (pd dataframe): A dataframe containing the train and test sets.
    """
    # index test and train participants into original dataframe
    train_participants = df_stratify[df_stratify['split']=='train']['Identifiers'].tolist()
    test_participants = df_stratify[df_stratify['split']=='test']['Identifiers'].tolist()

    df_train = df_original[df_original['Identifiers'].isin(train_participants)].reset_index(drop=True)
    df_train['split'] = 'train'

    df_test = df_original[df_original['Identifiers'].isin(test_participants)].reset_index(drop=True)
    df_test['split'] = 'test'

    # concat train and test
    df_out = pd.concat([df_train, df_test])

    return df_out


def run(
    fpath, 
    outpath,
    ):
    """Splits a Pandas DataFrame into train and test sets using stratified sampling, handling the case where some of the groups in the columns_to_stratify list have only 1 value.

    Args:
        fpath (str): Fullpath to dataframe to split, should be `all_participant_diagnoses.csv`
        outpath (str): Fullpath to file where the train/test sets will be saved. 
    Returns:
        df_out (pd dataframe): A dataframe containing the train and test sets.
    """
    import itertools
    import os
    import pandas as pd
    from hbn.data.data_utils import remove_small_groups

    columns_to_keep=['Sex', 'Age_round', 'PreInt_Demos_Fam,Child_Race_cat']
    columns_to_stratify = list(itertools.chain(*[columns_to_keep, ['DX_Index']]))
    test_size = 0.2
    random_state = 42

    # read in dataframe from path
    df = pd.read_csv(fpath, engine='python')

    # get unique dataset 
    df_unique = get_unique_dataset(df, 
        participant_col='Identifiers', 
        col_to_group='DX_Cat_Name', 
        cols_to_keep=columns_to_keep, 
        test_size=test_size, 
        random_state=random_state
        )
    
    # remove small groups from dataframe (otherwise won't be able to stratify)
    df_filtered = remove_small_groups(dataframe=df_unique, columns_to_stratify=columns_to_stratify)

    # stratify dataframe into train and test participants
    df_stratified = stratify_split(df_filtered, columns_to_stratify, test_size, random_state)

    # index train and test participants into original dataframe
    df_out = index_into_original_dataframe(df_original=df, df_stratify=df_stratified)

    # save dataframe to `out_dir`
    df_out.reset_index(drop=True).to_csv(outpath, index=False)
    print(f'train/test splits saved to {outpath}', flush=True)

    return df_out


if __name__ == "__main__":
    run()
    