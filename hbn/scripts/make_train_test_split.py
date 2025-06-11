
def stratify_split(df, columns_to_stratify=[], test_size=0.2, random_state=42):
    """Performs a stratified train-test split, handling rare combinations in the stratification columns.

    This function first fills NaN values in the specified stratification columns with 'Missing'.
    It then identifies unique combinations of the stratification columns that appear less than twice.
    The split is performed on the more frequent combinations using `train_test_split`.
    Finally, the rows with rare combinations are added entirely to the training set to avoid errors
    due to insufficient samples per class during stratified splitting.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame to split.
        columns_to_stratify (list, optional): A list of column names to use for stratified sampling.
            Defaults to an empty list, which results in a non-stratified split.
        test_size (float, optional): The proportion of the data to be included in the test set (0.0 to 1.0).
            Defaults to 0.2.
        random_state (int, optional): The random seed to use for shuffling the data before splitting.
            This ensures reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A new DataFrame that is the concatenation of the training and testing sets,
        with an added 'split' column indicating whether each row belongs to 'train' or 'test'.
    """
    from collections import Counter
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df_filled = df.copy()
    for col in columns_to_stratify:
        df_filled[col] = df_filled[col].fillna('Missing')

    if not columns_to_stratify:
        train_df, test_df = train_test_split(df_filled, test_size=test_size, random_state=random_state)
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        return pd.concat([train_df, test_df]).reset_index(drop=True)

    stratify_group = df_filled[columns_to_stratify].apply(tuple, axis=1)
    group_counts = Counter(stratify_group)
    rare_groups = {group for group, count in group_counts.items() if count < 2}
    df_rare = df_filled[stratify_group.isin(rare_groups)]
    df_common = df_filled[~stratify_group.isin(rare_groups)]

    stratify_common = df_common[columns_to_stratify]
    train_common, test_common = train_test_split(df_common, test_size=test_size, random_state=random_state, stratify=stratify_common)

    # Add split column and concatenate
    train_common['split'] = 'train'
    test_common['split'] = 'test'
    df_rare['split'] = 'train'  # Adding all rare cases to the training set

    df_out = pd.concat([train_common, test_common, df_rare]).reset_index(drop=True)

    return df_out


def get_unique_dataset(
    df,
    participant_col='Identifiers',
    col_to_group='DX_Cat_Name',
    cols_to_keep=['sex', 'age', 'race'],
    ):
    """Stratify split on unique participants, ensuring all participants are assigned a DX_Index.

    Args:
        df (pd dataframe): dataframe to stratify
    """
    import itertools
    import pandas as pd

    # get cols to keep, `columns_to_stratify` and `participant_col`
    cols_to_keep_with_part = [[participant_col], cols_to_keep]
    cols_to_keep_with_part = list(itertools.chain(*cols_to_keep_with_part))

    # Create a clean dataframe without NaNs in the grouping column
    df_clean = df.dropna(subset=[col_to_group])

    # Calculate the DX_Index based on the non-NaN groups
    dx_index = df_clean.groupby(col_to_group)[participant_col].count().sort_values().reset_index().reset_index().set_index(col_to_group)['index']

    # Map the DX_Index to all participants based on their first valid group
    def get_first_dx_index(series):
        first_valid_dx = series.dropna().iloc[0] if not series.dropna().empty else None
        return dx_index.get(first_valid_dx)

    df_with_first_dx = df.groupby(participant_col)[col_to_group].apply(get_first_dx_index).reset_index(name='DX_Index')

    # get unique dataset
    df_drop = df[cols_to_keep_with_part].drop_duplicates()
    df_unique = pd.merge(df_drop, df_with_first_dx, on=participant_col, how='left')

    return df_unique


def run(
    inpath, 
    outpath
    ):
    """Splits a Pandas DataFrame into train and test sets using stratified sampling, handling the case where some of the groups in the columns_to_stratify list have only 1 value.

    Args:
        inpath (str): Fullpath to dataframe to split, should be `all_participant_diagnoses.csv`
        outpath (str): Fullpath to file where the train/validate/test sets will be saved. 
    Returns:
        df_out (pd dataframe): A dataframe containing the train, validate, and test sets.
    """
    import os
    import itertools
    import numpy as np
    import pandas as pd

    columns_to_keep = ['sex', 'age_round', 'race'] 
    columns_to_stratify = list(itertools.chain(*[columns_to_keep, ['DX_Index']]))
    test_size = 0.2
    random_state = 42

    # read in dataframe from path
    df = pd.read_csv(inpath, engine='python')

    # get unique dataset 
    df_unique = get_unique_dataset(
        df, 
        participant_col='Identifiers', 
        col_to_group='DX_Cat_Name', 
        cols_to_keep=columns_to_keep, 
        )
    
    # stratify dataframe into train and test participants
    df_stratified = stratify_split(df_unique, columns_to_stratify, test_size, random_state)

    # index train and test participants into original dataframe
    df['age_round'] = df['age_round'].astype(object)
    df_out = df.merge(df_stratified, on=['Identifiers', 'sex', 'age_round', 'race'], how='outer')

    # replace 'missing' with NaN
    df_out = df_out.replace('Missing', np.nan)
    for col in ['age_round', 'DX_Index']:
        df_out[col] = df_out[col].astype(float)

    # save dataframe to `out_dir`
    df_out.to_csv(outpath, index=False)
    print(f'train/test splits saved to {outpath}', flush=True)

    return df_out


if __name__ == "__main__":
    run()
    