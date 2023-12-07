import os
from this import d
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import itertools
import glob
import re
import warnings

from hbn.data import make_dataset
from hbn import io
from hbn.constants import Defaults


def get_targets_NEW(
    binarize=True,
    participants=None,
    participant_groups=None
    ):
    """Return target dataframe

    Args:
        participants (list of str or None): (optional) if list of identifiers are passed, then returned dataframe filters for 'participants'
        participant_groups (list of str or None): (optional) group identifiers by disorder to allow for binarization/factorization
    Returns:
        dataframe (pd dataframe)
    """
    import pandas as pd

    # read in targets
    df = pd.read_csv(target_info['filename'])
    df = get_
    
    # optionally filter dataframe to contain certain participants
    if participants is not None:
        participants_df = pd.DataFrame(participants, columns=['Identifiers'])
        df = df.merge(participants_df, on='Identifiers')

    # change column values if participant_groups is given
    if participant_groups is not None:
        df['target'] = participant_groups
    
    # binarize `target`
    if binarize:
        df['target'] = df['target'].factorize()[0]

    # remove -1 (corresponds to "NaN")
    df = df[df['target']!=-1]

    df_target = df[['Identifiers', 'target']]

    return df_target

def get_targets(
    target_info,
    participants=None,
    participant_groups=None
    ):
    """Return target dataframe using arguments in `target_info` (data loaded from target spec file)

    Args:
        target_info (dict): dictionary loaded from target spec file (e.g., target_DX_<num>_Cat_binarize-spec.json)
        participants (list of str or None): (optional) if list of identifiers are passed, then returned dataframe filters for 'participants'
        participant_groups (list of str or None): (optional) group identifiers by disorder to allow for binarization/factorization
    Returns:
        dataframe (pd dataframe)
    """

    # get relevent features to predict
    # df = get_features(assessment=target_info['assessment'],
    #             domains=[target_info['domain']],
    #             measures=[target_info['measure']]
    #             )

    # read in targets
    df = pd.read_csv(target_info['filename'])
    
    # optionally filter dataframe to contain certain participants
    if participants is not None:
        participants_df = pd.DataFrame(participants, columns=['Identifiers'])
        df = df.merge(participants_df, on='Identifiers')

    # get variables from target_info
    col = target_info['target_column']
    target = target_info['transform']
    new_target = target_info['outname']

    # change column values if participant_groups is given
    if participant_groups is not None:
        df[col] = participant_groups
    
    # get new targets (binarize, factorize, or leave as is)
    if target=='binarize':
        df[new_target] = df[col].factorize()[0]
    elif target=='factorize':
        df[new_target] = df[col].factorize()[0]
    else:
        df[new_target] = df[col]

    # remove -1 (corresponds to "NaN")
    df = df[df[new_target]!=-1]

    df_target = df[['Identifiers', new_target]]

    return df_target


def preprocess(
        dataframe,
        cols_to_drop=['EID', 'Comment_ID', 'Administration', 'Days_Baseline', 'Data_entry', 'START_DATE', 'Year', 'Site', 'Season', 'Visit_label', 'Study', 'PSCID'],
        clf_info=None,
        cols_to_ignore=None,
        threshold=False
        ):

    """Preprocess the features (data cleaning, scaling, imputation, standarization, one-hot encoding)

    Args:
        dataframe (pd dataframe): pandas dataframe to preprocess, should include X features and y target var, output from `get_features`
        cols_to_drop (list of str): (optional) list of columns to drop from dataframe
        clf_info (dict of lists of scikit-learn classifiers or None): (optional) see `hbn/features/*.json` for example of structure. default is None
        cols_to_ignore (list of str or None): (optional) columns to ignore in preprocessing. Default is None.
        threshold (bool): threshold dataframe based on some fixed criterion. We are using 50% for columns and 20% for rows. If threshold is False, then only NaN entries are removed (no thresholding applied)
    """
    # do some scrubbing (e.g., remove superfluous columns)
    df_all = pd.DataFrame()
    for filter in cols_to_drop:
        df = dataframe.filter(like=filter)
        df_all = pd.concat([df_all, df], axis=1)

    # drop superfluous columns
    dataframe.drop(df_all.columns, axis=1, inplace=True)

    if threshold:
        # drop by threshold of NaN rows and columns 
        limitPerCols = dataframe.shape[1] * .50
        limitPerRows = dataframe.shape[0] * .20
        dataframe = dataframe.dropna(thresh=limitPerCols, axis='columns')
        dataframe = dataframe.dropna(thresh=limitPerRows, axis='index')

    # preprocessing: column transformation
    if clf_info is not None:
        dataframe = column_transform(dataframe=dataframe, clf_info=clf_info, cols_to_ignore=cols_to_ignore)
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]

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
        dataframe_to_ignore = dataframe[cols_to_ignore]

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
                # remainder='passthrough'
                )

    df_transformed = preprocesser.fit_transform(dataframe_final)

    # get transformed feature names (on fitted transformers only)
    feature_names = preprocesser.get_feature_names_out()

    # make pandas dataframe from transformed data
    df_transformed = pd.DataFrame(df_transformed, columns=feature_names)

    # add `col_to_ignore` back in
    if cols_to_ignore is not None:
        df_transformed = pd.concat([dataframe_to_ignore, df_transformed], axis=1)

    return df_transformed


def drop_duplicates(dataframe):
    """some measures (i.e. Teacher) have duplicate rows (multiple Identifiers). We take the mean across the duplicate Identifiers (for numeric columns)
    and take the first row (for object columns)
    Args:
        dataframe (pd dataframe):
    Returns: 
        df (pd dataframe): preprocessed dataframe (remove duplicates)
    """
    # remove duplicate columns
    dataframe = dataframe.loc[:,~dataframe.columns.duplicated()].copy()
    # remove trailing numbers (e.g., '_1', '_2')
    dataframe['Identifiers'] = dataframe['Identifiers'].str.split('_').str.get(0)
    # group by unique identifiers and take the mean value for the numeric items
    tmp = dataframe.groupby('Identifiers').mean(numeric_only=True).reset_index()
    # group by unique identifiers and take the first row of the object items
    tmp2 = dataframe.select_dtypes(include='object').groupby('Identifiers').first().reset_index()
    # merge both the numeric and object dataframes together
    df = tmp.merge(tmp2, on='Identifiers')

    return df


def smote(y_train, X_train):
    """oversamples `y_train` and `X_train` for minority samples

    Args: 
        y_train (pd dataframe):
        X_train (pd dataframe):
    Returns:
        df_smote (pd dataframe)
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler

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


