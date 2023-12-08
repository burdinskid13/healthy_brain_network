from src import io
import pandas as pd 
import numpy as np
import os

def get_features(info, dirn):
    """Create features from parameters set in `feature_spec`, do some basic preprocessing (removing superfluous columns)

    Args:
        info (dict): dictionary loaded from `feature_spec`
        dirn (str): directory where 'filename' from `feature_spec` is located
    Retunrs:
        df (pd dataframe): dataframe of features
    """

    # load in csv file
    df = pd.read_csv(os.path.join(dirn, info['filename']))

    # do some scrubbing (e.g., remove superfluous columns)
    df_drop = pd.DataFrame()
    for filter in info['cols_to_drop']:
        df_filter = df.filter(like=filter)
        df_drop = pd.concat([df_drop, df_filter], axis=1)
    df.drop(df_drop.columns, axis=1, inplace=True)

    return df


def get_targets(info, dirn):
    """Create target(s) from parameters set in `target_spec`, do some basic preprocessing (binarize `target_column` and impute if there are NaN values)

    Args:
        info (dict): dictionary loaded from `target_spec`
        dirn (str): directory where 'filename' from `target_spec` is located
    Retunrs:
        df (pd dataframe): dataframe of targets. 
    """

    # load in csv file
    df = pd.read_csv(os.path.join(dirn, info['filename']))

    # which column is the target
    target = info['target_column']

    # which columns are we keeping
    if len(info['cols_to_keep']) > 0:
        df = df[info['cols_to_keep']]

    # binarize `target_column`
    if info['binarize']:
        df[target] = df[target].factorize()[0]
        # remove -1 (corresponds to "NaN")
        df = df[df[target]!=-1]

    return df


def get_participants(info, dirn):
    """Get participant identifiers from parameters set in `participant_spec`

    Args:
        info (dict): dictionary loaded from `participant_spec`
        dirn (str): directory where 'filename' from `participant_spec` is located
    Retunrs:
        df (pd dataframe): dataframe of participant identifiers 
    """

    # load in csv file
    df = pd.read_csv(os.path.join(dirn, info['filename']))

    # get columns and values from variables set in `participant_spec`
    columns = list(info.keys())
    values = list(info.values())

    # return relevant participants indexed by columns and values
    df = _index_dataframe_by_columns_values(dataframe=df, 
                                    columns=columns, 
                                    values_list=values
                                    )

    return df


def combine_features_and_targets(features, targets, participant_id, participants=None):
    """Combine features and targets into a single dataframe merging on `participant_id`. 
    Optionally index by `participant_id` present in `participants`

    Args:
        features (pandas.DataFrame): The features dataframe.
        targets (pandas.DataFrame): The targets dataframe.
        participant_id (str): The column name of the participant identifier. Should be present in `features`, `targets`, and `participants`.
        participants (pandas.DataFrame or None): The participants dataframe. Default is None.
    Returns:
        pandas.DataFrame: The combined and filtered dataframe.
    """

    # Check if the number of columns in the features and targets dataframes match.
    # if features.shape[0] != targets.shape[0]:
    #     raise ValueError("The number of rows in the features and targets dataframes must match.")

    # get the intersection of columns between dataframes and remove `participant_id`
    common_cols = list(features.columns.intersection(targets.columns))
    common_cols = [c for c in common_cols if participant_id!=c]

    # make sure the target column is not included in features dataframe
    features = features.drop(common_cols, axis=1)

    # Combine the features and targets dataframes into a single dataframe.
    combined_df = features.merge(targets)

    # drop duplicates
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    # Index the combined dataframe by `participant_id` in participants dataframe (if provided).
    if participants is not None:
        participants_list = participants[participant_id].tolist()
        combined_df = combined_df[combined_df[participant_id].isin(participants_list)]

    return combined_df


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
            row_matches.append(dataframe[column].isin(values_list[i]))

    # if there are multiple columns, combine the boolean values into a single boolean value
    if len(row_matches)>0:
        if len(row_matches)>1:
            row_matches = np.logical_and(*row_matches)
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


def smote(y_train, X_train):
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


def preprocess(
        dataframe,
        clf_info=None,
        cols_to_ignore=None,
        cols_to_drop=None,
        threshold=False,
        upsample=True,
        target_column=None
        ):

    """Preprocess the features (data cleaning, scaling, imputation, standarization, one-hot encoding)

    Args:
        dataframe (pd dataframe): pandas dataframe to preprocess, should include X features and y target var, output from `get_features`
        clf_info (dict of lists of scikit-learn classifiers or None): (optional) see `base_specs.features` for an example.
        cols_to_ignore (list of str or None): (optional) columns to ignore in preprocessing. Default is None.
        cols_to_drop (list of str or None): (optional) columns to drop in preprocessing. Default is None.
        threshold (bool): threshold dataframe based on some fixed criterion. We are using 50% for columns and 20% for rows. If threshold is False, then only NaN entries are removed (no thresholding applied)
        upsample (bool): upsample minority class using SMOTE. default is True
        target_column (str): target column name. default is None
    """

    if threshold:
        # drop by threshold of NaN rows and columns 
        limitPerCols = dataframe.shape[1] * .50
        limitPerRows = dataframe.shape[0] * .20
        dataframe = dataframe.dropna(thresh=limitPerCols, axis='columns')
        dataframe = dataframe.dropna(thresh=limitPerRows, axis='index')

    # preprocessing: column transformation
    if clf_info is not None:
        dataframe = column_transform(
            dataframe=dataframe, 
            clf_info=clf_info, 
            cols_to_ignore=cols_to_ignore
            )
    
    # optionally drop features
    for col in cols_to_drop:
        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains(col)]

    # upsample minority target class using smote 
    if upsample and target_column:
        x_cols = [col for col in dataframe.columns if target_column not in col]
        dataframe = smote(y_train=dataframe[[target_column]], X_train=dataframe[x_cols])

    return dataframe