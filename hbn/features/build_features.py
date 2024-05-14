import pandas as pd 
import numpy as np
import os

from hbn import io

def get_features(info, dirn):
    """Create features from parameters set in `feature_spec`, do some basic preprocessing (removing superfluous columns)

    Args:
        info (dict): dictionary loaded from `feature_spec`
        dirn (str): directory where 'filename' from `feature_spec` is located
    Retunrs:
        df (pd dataframe): dataframe of features
    """

    # load in csv file
    df = pd.read_csv(os.path.join(dirn, info['filename']), engine='python')

    # do some scrubbing (e.g., remove superfluous columns)
    df_drop = pd.DataFrame()
    for filter in info['cols_to_drop']:
        df_filter = df.filter(like=filter)
        df_drop = pd.concat([df_drop, df_filter], axis=1)

    df_out = df.drop(df_drop.columns, axis=1)

    # filter columns if that is specified
    if info['cols_to_filter'] is not None:
        df_all = pd.DataFrame()
        for col in info['cols_to_filter']:
            df_filter = df_out.loc[:, df_out.columns.str.contains(col)]
            df_all = pd.concat([df_filter, df_all], axis=1)

        return df_all.drop_duplicates().reset_index(drop=True)
    else:
        return df_out.drop_duplicates().reset_index(drop=True)


def get_targets(info, dirn):
    """Create target(s) from parameters set in `target_spec`, do some basic preprocessing (binarize `target_column` and impute if there are NaN values)

    Args:
        info (dict): dictionary loaded from `target_spec`
        dirn (str): directory where 'filename' from `target_spec` is located
    Retunrs:
        df (pd dataframe): dataframe of targets. 
    """

    # load in csv file
    df = pd.read_csv(os.path.join(dirn, info['filename']), engine='python')

    # which column is the target
    target = info['target_column']

    # which columns are we keeping
    if len(info['cols_to_keep']) > 0:
        df = df[info['cols_to_keep']]

    return df.reset_index(drop=True)


def get_participants(info, dirn):
    """Get participant identifiers from parameters set in `participant_spec`

    Args:
        info (dict): dictionary loaded from `participant_spec`
        dirn (str): directory where 'filename' from `participant_spec` is located
    Retunrs:
        df (pd dataframe): dataframe of participant identifiers 
    """

    # load in csv file
    df = pd.read_csv(os.path.join(dirn, info['filename']), engine='python')

    # get columns and values from variables set in `participant_spec`
    columns = list(info.keys())
    values = list(info.values())

    # return relevant participants indexed by columns and values
    df = _index_dataframe_by_columns_values(dataframe=df, 
                                    columns=columns, 
                                    values_list=values
                                    )

    return df.reset_index(drop=True)


def combine_features_and_targets(features, targets, merge_on):
    """Combine features and targets into a single dataframe merging on `participant_id`. 

    Args:
        features (pandas.DataFrame): The features dataframe.
        targets (pandas.DataFrame): The targets dataframe.
        merge_on (str): The column name of the merge key (e.g., 'Identifiers' or 'participant_id') Should be present in `features`, `targets`, and `participants`.
    Returns:
        pandas.DataFrame: The combined and filtered dataframe.
    """

    # Check if the number of columns in the features and targets dataframes match.
    # if features.shape[0] != targets.shape[0]:
    #     raise ValueError("The number of rows in the features and targets dataframes must match.")

    # get the intersection of columns between dataframes and remove `participant_id`
    common_cols = list(features.columns.intersection(targets.columns))
    common_cols = [c for c in common_cols if merge_on!=c]

    # make sure common cols are not included in features dataframe
    features_drop_common_cols = features.drop(common_cols, axis=1)

    # Combine the features and targets dataframes into a single dataframe.
    combined_df = features_drop_common_cols.merge(targets, on=merge_on)

    return combined_df


def merge_with_participants(dataframe, participants, participant_id, merge_cols=['Identifiers', 'DX_Cat_Name']):
    """ filter `dataframe` with participants dataframe on `participant_id` and `merge_cols`

    Args:
        dataframe (pandas.DataFrame): The dataframe to filter.
        participants (pandas.DataFrame): The participants dataframe.
        participant_id (str): The column name of the participant identifier.
        merge_cols (list): The columns to merge on.
    Returns:
        df_merged (pandas.DataFrame): The filtered dataframe.
    """
    # list out all participants
    participants_list = participants[participant_id].tolist()

    # filter dataframe on those participants
    df_filter_participants = dataframe[dataframe[participant_id].isin(participants_list)].reset_index(drop=True)

    # combine features, targets, and new participant dataframe
    df_merged = participants.merge(df_filter_participants, on=merge_cols)

    return df_merged


def _index_dataframe_by_columns_values(dataframe, columns, values_list):
    """Indexes a Pandas DataFrame using a list of columns and a list of specific values.

    Args:
    dataframe (pandas.DataFrame): The DataFrame to index.
    columns (list): A list of column names to index by.
    values_list (list): A list of specific values to match.

    Returns:
    pandas.DataFrame: The filtered DataFrame.
    """

    # Check if the number of columns and values in the lists match.
    if len(columns) != len(values_list):
        raise ValueError("The number of columns and values in the lists must match.")

    # Create a list of boolean values indicating whether each row matches the values.
    row_matches = []
    for i, column in enumerate(columns):
        if column in dataframe.columns:
            row_matches.append(list(dataframe[column].isin(values_list[i])))

    # if there are multiple columns, combine the boolean values into a single boolean value
    if len(row_matches)>0:
        if len(row_matches)>1:
            row_matches = np.all(row_matches, axis=0)
        else:
            row_matches = row_matches[0]

        # Filter the DataFrame to only include rows where the row_matches_combined value is True.
        dataframe = dataframe[row_matches]

    return dataframe


def column_transform(
    dataframe,
    clf_info,
    cols_to_ignore=None,
    ):
    """Column Transformation on `dataframe` using classifier information passed in by `clf_info`, `cols_to_ignore` in dataframe are ignored

    Args: 
        dataframe (pd dataframe): pandas dataframe, `cols_to_ignore` should be in `dataframe`. output from `get_features`
        clf_info (dict of classifier): example is {"numeric": [["sklearn.impute", "SimpleImputer", {"strategy": "mean"}], ["sklearn.preprocessing", "StandardScaler", {}]]}
        cols_to_ignore (list of str or None): default is None.
    Returns:
        `df_transformed` (pd dataframe): first columns are `cols_to_ignore` if they are not None.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector as selector
    from sklearn.utils.validation import check_is_fitted
    import pandas as pd

    ## functionality borrowed from pydra-ml
    def to_instance(clf_info):
        mod = __import__(clf_info[0], fromlist=[clf_info[1]])
        params = {}
        if len(clf_info) > 2:
            params = clf_info[2]
        clf = getattr(mod, clf_info[1])(**params)
        if len(clf_info) == 4:
            from sklearn.model_selection import GridSearchCV

            clf = GridSearchCV(clf, param_grid=clf_info[3])
        return clf

    def make_pipeline(clf_info):
        if isinstance(clf_info[0], list):
            # Process as a pipeline constructor
            steps = []
            for val in clf_info:
                step = to_instance(val)
                steps.append((val[1], step))
            pipe = Pipeline(steps)
        else:
            clf = to_instance(clf_info)
            from sklearn.preprocessing import StandardScaler
            pipe = Pipeline([("std", StandardScaler()), (clf_info[1], clf)])
        return pipe

    # drop `cols_to_ignore`
    dataframe_final = pd.DataFrame()
    if cols_to_ignore is not None:
        dataframe_final = dataframe.drop(cols_to_ignore, axis=1)
        dataframe_to_ignore = dataframe[cols_to_ignore].reset_index(drop=True)

    # set up numeric pipeline
    transformers = []
    for key in clf_info.keys():
        pipe = make_pipeline(clf_info=clf_info[key])
        if key == 'numeric':
            transformers.append((key, pipe, selector(dtype_include="number")))
        elif key == 'category':
            transformers.append((key, pipe, selector(dtype_exclude="number")))

    # column transformer
    preprocesser = ColumnTransformer(transformers=transformers,
                verbose_feature_names_out=True,
                #remainder='passthrough'
                )

    arr_transformed = preprocesser.fit_transform(dataframe_final)

    # get transformed feature names (on fitted transformers only)
    feature_names = preprocesser.get_feature_names_out()

    # make pandas dataframe from transformed data
    df_transformed = pd.DataFrame(arr_transformed, columns=feature_names)

    # add `col_to_ignore` back in
    if cols_to_ignore is not None:
        df_transformed = pd.concat([dataframe_to_ignore, df_transformed], axis=1)

    return df_transformed


def upsample_data(y_train, X_train):
    """oversamples `y_train` and `X_train` for minority samples

    Args: 
        y_train (pd dataframe):
        X_train (pd dataframe):
    Returns:
        df_smote (pd dataframe)
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    import pandas as pd
    import numpy as np

    # try SMOTE and if it throws an error, try RandomOverSampler to oversample the minority class
    try:
        sm = SMOTE(random_state=42, sampling_strategy='auto') # was .5
        X_train_oversampled, y_train_oversampled = sm.fit_resample(np.array(X_train), np.array(y_train))
    except:
        ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
        X_train_oversampled, y_train_oversampled = ros.fit_resample(np.array(X_train), np.array(y_train))

    new_x = pd.DataFrame(X_train_oversampled, columns=X_train.columns)
    new_y = pd.DataFrame(y_train_oversampled, columns=y_train.columns)
    df_smote = pd.concat([new_x, new_y], axis=1)
    return df_smote


def _drop_long_string_columns(df):
    """
    Checks the string length of each column in a pandas dataframe and drops columns that have string lengths longer than 259 characters.

    Args:
        df (pandas.DataFrame): The pandas dataframe to be processed.

    Returns:
        pandas.DataFrame: The pandas dataframe with columns that have string lengths longer than 259 characters dropped.
    """
    import pandas as pd

    long_str_cols = []
    for col in df.columns:
        if len(col)>259:
            long_str_cols.append(col)

    df = df.drop(columns=long_str_cols)
    return df


def remove_mixed_nan_zero_columns(df):
    import numpy as np
    import pandas as pd
    def is_mixed_nan_zero(col):
        if all(isinstance(val, (float, np.float64)) and val==0 for val in col):
            return True
        elif all(pd.isna(val) for val in col):
            return True
        else:
            return False
        
    # identify cols to remove
    cols_to_remove = [col for col in df.columns if is_mixed_nan_zero(df[col])]

    # remove the identified cols
    df = df.drop(cols_to_remove, axis=1)
    return df


def preprocess(
        dataframe,
        clf_info=None,
        cols_to_ignore=None,
        cols_to_drop=None,
        threshold=False,
        target_column=None,
        binarize_target=True
        ):

    """Preprocess the features (data cleaning, scaling, imputation, standarization, one-hot encoding)

    Args:
        dataframe (pd dataframe): pandas dataframe to preprocess, should include X features and y target var, output from `get_features`
        clf_info (dict of lists of scikit-learn classifiers or None): (optional) see `base_specs.features` for an example.
        cols_to_ignore (list of str or None): (optional) columns to ignore in preprocessing. Default is None.
        cols_to_drop (list of str or None): (optional) columns to drop in preprocessing. Default is None.
        threshold (bool): threshold dataframe based on some fixed criterion. We are using 50% for columns and 20% for rows. If threshold is False, then only NaN entries are removed (no thresholding applied)
        target_column (str): target column name. default is None
        binarize_target (bool): binarize target column if target_column is not None. default is True
    """

    if threshold:
        # drop by threshold of NaN rows and columns 
        limitPerCols = dataframe.shape[1] * .50
        limitPerRows = dataframe.shape[0] * .20
        dataframe = dataframe.dropna(thresh=limitPerCols, axis='columns')
        dataframe = dataframe.dropna(thresh=limitPerRows, axis='index')

    # preprocessing on features: column transformation
    if clf_info is not None:
        dataframe = column_transform(
            dataframe=dataframe, 
            clf_info=clf_info, 
            cols_to_ignore=cols_to_ignore
            )

    # preprocessing on target: binarize `target`
    if binarize_target:
        dataframe[target_column] = dataframe[target_column].factorize()[0]
        # remove -1 (corresponds to "NaN")
        dataframe = dataframe[dataframe[target_column]!=-1]
    
    # optionally drop features
    for col in cols_to_drop:
        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains(col)]

    # drop features that are too long (longer than 259 characters) - causes errors when we run pydraml
    dataframe = _drop_long_string_columns(df=dataframe)

    # delete columns that are all NaN or 0
    dataframe = remove_mixed_nan_zero_columns(df=dataframe)

    return dataframe