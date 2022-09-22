from codecs import ascii_decode
import os
import numpy as np
import pandas as pd
import glob
import re
import warnings

from hbn.data import make_dataset
from hbn.constants import Defaults


def get_data(
    assessment='Child Measures',
    domains='all',
    measures='all',
    target='CGAS_Score',
    min_num_participants=2000,
    incl_data_type=None
    ):
    """read in data from `data/raw/phenotype/Assessment_List_Jan2019.xlsx`:
    `assessment` (e.g., Child Measures, Parent Measures), `domains` (e.g., Cognitive Testing), `measures` (e.g., Kaufman Brief Intelligence Test-II)

    Args: 
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Teacher Measures', 'Clinical Measures'. Default is 'Child Measures'
        domains (list of str or 'all'): exhaustive list, find options here: `data/raw/phenotype/Assessment_List_Jan2019.xlsx`. Default is 'all'. If `domains` is 'all', all domains are loaded
        measures (list of str or 'all'): exhaustive list, find options here: `data/raw/phenotype/Assessment_List_Jan2019.xlsx`. Default is 'all'. If `measures` is 'all', all measures are loaded for `domains`
        target (str or None): default is 'CGAS_Score'. other options: 'DX_01_factorize', 'DX_01_Cat_factorize', 'DX_01_binary', 'Sex_binarize'. If None, doesn't include `target` column
        min_num_participants (int): min_num_participants for inclusion of assessment/domain/measure as features
        incl_data_type (list of pd.DataFrame.dtypes or None): if None, all categories are returned. default is None. pd.DataFrame.dtypes options: 'number', 'float', 'int', 'datetime', 'object'
    """

    # check input args - `domains` and `measures` must be list or None
    if (isinstance(domains, str)) and (domains != 'all'):
        domains = [domains]
    if (isinstance(measures, str)) and (measures != 'all'):
        measures = [measures]

    # get directory
    assessment = '_'.join(re.split(r'_|,|/| ', assessment))
    fdir = os.path.join(Defaults.PHENO_DIR, assessment) # join with '_'

    # load in participants
    _, identifiers = make_dataset.get_clinical_diagnosis(demographics=False, target=target)

    # get domains
    if domains == 'all':
        domain_dir = glob.glob(os.path.join(fdir, '*'))
    else:
        domain_dir = [os.path.join(fdir, '_'.join(re.split(r'_|,|/| ', d))) for d in domains]

    # loop over domains
    df_all = pd.DataFrame({'Identifiers': identifiers})
    for domain in domain_dir:

        # get measures
        if measures == 'all':
            measure_dir = glob.glob(os.path.join(domain, '*'))
        else:
            measure_dir = [os.path.join(domain, '_'.join(re.split(r'_|,|/| ', m)) + '.csv') for m in measures] # join with '_'

        # loop over measures
        for measure in measure_dir:
            df = pd.read_csv(measure)

            if len(df)>=min_num_participants:
                df_all = df_all.merge(df, on='Identifiers')
                print(f'reading {measure} into dataframe')
            else:
                print(f'fewer than {min_num_participants} in {measure}, not included as features')

    # add clinical + demographic info as `target`
    if 'CGAS' in target:
        df_all = make_dataset._add_CGAS_Score(dataframe=df_all)
    elif 'DX' in target:
        dx, _ = make_dataset.get_clinical_diagnosis(demographics=False, target=target)
        df_all = dx[['Identifiers', target]].merge(df_all, on='Identifiers')
    elif 'Sex' in target:
        dx = make_dataset._add_demographics(dataframe=df_all)
        df_all = dx[[target, 'Identifiers']].merge(df_all, on='Identifiers')

    # drop NaN
    df_all = df_all.replace(' ', np.float("NaN")).fillna(np.float("NaN")).dropna(how='all', axis=1)
    df_all = df_all.dropna(how='all', axis=0)

    if incl_data_type is not None:
        df_all = df_all.select_dtypes(include=incl_data_type)

    return df_all

def preprocess(
        dataframe,
        cols_to_drop=['Identifiers', 'EID', 'Comment_ID', 'Administration', 'Days_Baseline', 'Data_entry', 'START_DATE', 'Year', 'Site', 'Season', 'Visit_label', 'Study', 'PSCID'],
        clf_info=None,
        cols_to_ignore=['DX_01_Cat_factorize']
        ):

    """Preprocess the features (data cleaning, scaling, imputation, standarization, one-hot encoding)

    Args:
        dataframe (pd dataframe): pandas dataframe to preprocess, should include X features and y target var
        cols_to_drop (list of str): list of columns to drop from dataframe
        clf_info (dict of lists of scikit-learn classifiers): see `hbn/features/features-example.json` for example of structure
        cols_to_ignore (list of str): columns to ignore in preprocessing
    """
    
    # do some scrubbing (e.g., remove superfluous columns)
    df_all = pd.DataFrame()
    for filter in cols_to_drop:
        df = dataframe.filter(like=filter)
        df_all = pd.concat([df_all, df], axis=1)

    # drop superfluous columns
    dataframe.drop(df_all.columns, axis=1, inplace=True)

    # drop any all NaN rows
    dataframe = dataframe.dropna(how='all', axis=0)
    dataframe = dataframe.dropna(how='all', axis=1)

    # preprocessing: column transformation
    if clf_info is not None:
        dataframe = column_transform(dataframe=dataframe, clf_info=clf_info, cols_to_ignore=cols_to_ignore)

    return dataframe

def column_transform(
    dataframe,
    clf_info,
    cols_to_ignore=None,
    ):
    """Column Transformation on `dataframe` using classifier information passed in by `clf_info`, `cols_to_ignore` in dataframe are ignored

    Args: 
        dataframe (pd dataframe): pandas dataframe, `cols_to_ignore` should be in `dataframe`
        clf_info (dict of classifier): example is {"numeric": [["sklearn.impute", "SimpleImputer", {"strategy": "mean"}], ["sklearn.preprocessing", "StandardScaler", {}]]}
        cols_to_ignore (list of str or None): default is None.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector as selector

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
    dataframe_to_ignore = pd.DataFrame()
    if cols_to_ignore is not None:
        dataframe_dropped = dataframe.drop(cols_to_ignore, axis=1)
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

    # transform the data
    df_transformed = preprocesser.fit_transform(dataframe_dropped)

    # get transformed feature names
    feature_names = get_feature_names(column_transformer=preprocesser)

    # make pandas dataframe from transformed data
    df_transformed = pd.DataFrame(df_transformed, columns=feature_names)

    # add `col_to_ignore` back in
    df_transformed = pd.concat([dataframe_to_ignore, df_transformed], axis=1)

    return df_transformed

def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    from sklearn.pipeline import Pipeline
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

def get_domains(assessment='Child Measures'):
    """get domains for `assessment`

    Args:
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'
    Returns:
        list of str: list of domains
    """
    fname = '_'.join(assessment.split())

    # master info file
    fpath = os.path.join(Defaults.PHENO_DIR, f'Assessment_List_Jan2019_{fname}.csv')

    if not os.path.isfile(fpath):
        make_dataset.assessment_list(assessment=assessment)
    
    # read in corrected assessment list
    info = pd.read_csv(fpath)

    if 'Domain' in info.columns:
        domains = info['Domain'].unique()
    elif 'Measure' in info.columns:
        domains = info['Measure'].unique()

    return {assessment: domains}
