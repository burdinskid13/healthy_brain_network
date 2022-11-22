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


def get_features(
    assessment='Child Measures',
    domains='all',
    measures='all',
    min_num_participants=2000,
    incl_data_type=None
    ):
    """read in data from `data/raw/phenotype/Assessment_List_Jan2019.xlsx`:
    `assessment` (e.g., Child Measures, Parent Measures), `domains` (e.g., Cognitive Testing), `measures` (e.g., Kaufman Brief Intelligence Test-II)

    Args: 
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Teacher Measures', 'Clinical Measures'. Default is 'Child Measures'
        domains (list of str or 'all'): exhaustive list, find options here: `data/raw/phenotype/Assessment_List_Jan2019.xlsx`. Default is 'all'. If `domains` is 'all', all domains are loaded
        measures (list of str or 'all'): exhaustive list, find options here: `data/raw/phenotype/Assessment_List_Jan2019.xlsx`. Default is 'all'. If `measures` is 'all', all measures are loaded for `domains`
        min_num_participants (int): min_num_participants for inclusion of assessment/domain/measure as features
        incl_data_type (list of pd.DataFrame.dtypes or None): if None, all categories are returned. default is None. pd.DataFrame.dtypes options: 'number', 'float', 'int', 'datetime', 'object'
    Returns:
        df_all (pd dataframe)
    """

    # check input args - `domains` and `measures` must be list or None
    if (isinstance(domains, str)) and ('all' in domains):
        domains = [domains]
    if (isinstance(measures, str)) and ('all' in measures):
        measures = [measures]

    # get directory
    assessment = '_'.join(re.split(r'_|,|/| ', assessment))
    fdir = os.path.join(Defaults.PHENO_DIR, assessment) # join with '_'

    # get domains
    if None in domains:
        domain_dir = [fdir]
    elif 'all' in domains:
        domain_dir = glob.glob(os.path.join(fdir, '*'))
    else:
        domain_dir = [os.path.join(fdir, '_'.join(re.split(r'_|,|/| ', d))) for d in domains]

    # load in participants
    participants_fpath = os.path.join(Defaults.PHENO_DIR, 'participants.csv')
    if not os.path.isfile(participants_fpath):
        make_dataset.get_summary()
    identifiers = pd.read_csv(participants_fpath)['Identifiers']

    # loop over domains
    df_all = pd.DataFrame({'Identifiers': identifiers})
    for domain in domain_dir:

        # get measures
        if 'all' in measures:
            measure_dir = glob.glob(os.path.join(domain, '*'))
        else:
            measure_dir = [os.path.join(domain, '_'.join(re.split(r'_|,|/| ', m)) + '.csv') for m in measures] # join with '_'

        # loop over measures
        for measure in measure_dir:
            
            # only read in files that exist
            if os.path.isfile(measure):
                df = pd.read_csv(measure)

                if len(df)>=min_num_participants:
                    df_all = df_all.merge(df, on='Identifiers')
                    print(f'reading {measure} into dataframe')
                else:
                    logger = _setup_logger('first_logger', 'too-few-features.log')
                    logger.info(Path(measure).name)
            else:
                super_logger = _setup_logger('second_logger', 'features-nonexistent.log')
                super_logger.info(Path(measure).name)

    # drop NaN
    df_all = df_all.replace(' ', np.float("NaN")).fillna(np.float("NaN")).dropna(how='all', axis=1)
    df_all = df_all.dropna(how='all', axis=0)

    if incl_data_type is not None:
        df_all = df_all.select_dtypes(include=incl_data_type)

    return df_all


def get_targets(
    target_info,
    participants=None
    ):
    """Return target dataframe using arguments in `target_info` (data loaded from target spec file)

    Args:
        target_info (dict): dictionary loaded from target spec file (e.g., target_DX_01_Cat_binarize-spec.json)
        participants (list of str or None): (optional) if list of identifiers are passed, then returned dataframe filters for 'participants'

    Returns:
        dataframe (pd dataframe)
    """

    def _binarize_diagnosis(x):
        if 'No Diagnosis Given' in x:
            return 0
        else:
            return 1

    # get questionnaire
    df = get_features(assessment=target_info['assessment'],
                domains=[target_info['domain']],
                measures=[target_info['measure']]
                )

    # optionally filter dataframe to contain certain participants
    if participants is not None:
        participants_df = pd.DataFrame(participants, columns=['Identifiers'])
        df = df.merge(participants_df, on='Identifiers')

    # category of target
    col = target_info['target_column']
    target = target_info['transform']
    new_target = target_info['outname']
    
    # do some cleanup
    if 'DX' in col:
        df[col] = df[col].fillna('No Diagnosis Given')

    if target=='binarize':
        if 'DX' in col:
            df[new_target] = df[col].apply(lambda x: _binarize_diagnosis(x))
        else:
            df[new_target] = df[col].factorize()[0]
    elif target=='factorize':
        df[new_target] = df[col].factorize()[0]
    else:
        df[new_target] = df[col]

    df_target = df[['Identifiers', new_target]]

    return df_target


def preprocess(
        dataframe,
        cols_to_drop=['EID', 'Comment_ID', 'Administration', 'Days_Baseline', 'Data_entry', 'START_DATE', 'Year', 'Site', 'Season', 'Visit_label', 'Study', 'PSCID'],
        clf_info=None,
        cols_to_ignore=['DX_01_Cat_factorize']
        ):

    """Preprocess the features (data cleaning, scaling, imputation, standarization, one-hot encoding)

    Args:
        dataframe (pd dataframe): pandas dataframe to preprocess, should include X features and y target var, output from `get_features`
        cols_to_drop (list of str): list of columns to drop from dataframe
        clf_info (dict of lists of scikit-learn classifiers or None): see `hbn/features/features-example.json` for example of structure
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

    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]

    return dataframe


def make_feature_specs(parent_spec, out_dir=Defaults.FEATURE_DIR):
    """make feature sets (json spec files)

    Args: 
        parent_spec (str): full path to master spec file. saved in `out_dir`
        out_dir (str): save to path. default is `Defaults.FEATURE_DIR`
    Returns:
        saves feature spec files (.json) to `FEATURE_DIR` and returns list of feature specs
    """
    from hbn import io

    feature_combinations = _get_feature_combinations(parent_spec)
    parent_spec_info = io.read_json(parent_spec)

    def _make_filename(data):
        """make filename for feature spec

        Args: 
            data (dict):
        Returns:
            spec_file (str): spec filename
        """
        # define spec filename
        vals = []
        for val in ['features', 'assessment', 'domains', 'measures']:
            if val in data.keys():
                vals.append('_'.join(re.split(r'_|,|/| ', data[val])))
            else:
                vals.append(val)
        spec_file = '-'.join(vals)
        return spec_file
    
    spec_files = []
    for data in feature_combinations:
        
        spec_filename = _make_filename(data)

        # define feature spec file
        spec_info = {
                    # "filename": spec_filename + '.csv', 
                    "assessment": data['assessment'],
                    "domains": data['domains'],
                    "measures": data['measures'],
                    "preprocessing": parent_spec_info['preprocessing'], 
                    "min_num_participants": parent_spec_info['min_num_participants']
                    }

        # save json to `FEATURE_DIR`
        spec_fpath = os.path.join(out_dir, spec_filename + '-spec.json')
        io.save_dict_as_JSON(fpath=spec_fpath, data_dict=spec_info)
        print(f'spec file and features saved to disk for {spec_filename}')
        spec_files.append(spec_fpath)

    return spec_files


def make_target_specs(parent_spec, out_dir=Defaults.FEATURE_DIR):
    """make target sets (json spec files)

    Args: 
        parent_spec (str): full path to master spec file. saved in `out_dir`
        out_dir (str): save to path. default is `Defaults.FEATURE_DIR`
    Returns:
        saves feature spec files (.json) to `FEATURE_DIR` and returns list of feature specs
    """
    from hbn import io

    parent_spec_info = io.read_json(parent_spec)

    targets = parent_spec_info['data']["target"]

    spec_files = []
    for data in targets:
        
        spec_filename = 'target_' + data["outname"]

        # define target spec file
        spec_info = {
                    # "filename":  spec_filename +'.csv',
                    "assessment": data["assessment"],
                    "domain": data["domain"],
                    "measure": data["measure"],
                    "target_column": data["target_column"],
                    "transform": data["transform"], 
                    "outname": data["outname"],
                    "preprocessing": parent_spec_info['preprocessing'], 
                    "min_num_participants": parent_spec_info['min_num_participants']
                    }

        # save json to `FEATURE_DIR`
        spec_fpath = os.path.join(out_dir, spec_filename + '-spec.json')
        io.save_dict_as_JSON(fpath=spec_fpath, data_dict=spec_info)
        print(f'spec file and features saved to disk for {spec_filename}')
        spec_files.append(spec_fpath)

    return spec_files


def make_parent_spec(out_dir=Defaults.FEATURE_DIR):
    """make parent spec files (.json file) - all possible combinations of feature specs
    """

    spec_info = {   
            "data": {
                "features": {
                    "assessment": ["Child Measures", "Parent Measures", "Teacher Measures"],
                    "domains": "all",
                    "measures": "all"
                    },
                "target": [
                        {"assessment": "Clinical Measures",
                        "domain": None,
                        "measure": "Clinical Diagnosis Demographics",
                        "target_column": "DX_01_Cat",
                        "transform": "binarize",
                        "outname": "DX_01_Cat_binarize"
                        },
                        {"assessment": "Clinical Measures",
                        "domain": None,
                        "measure": "Clinical Diagnosis Demographics",
                        "target_column": "DX_01_Cat",
                        "transform": "factorize",
                        "outname": "DX_01_Cat_factorize"
                        },
                        {"assessment": "Clinical Measures",
                        "domain": None,
                        "measure": "Clinical Diagnosis Demographics",
                        "target_column": "DX_01",
                        "transform": "binarize",
                        "outname": "DX_01_binarize"
                        }
                        # {"assessment": "Clinical Measures",
                        # "domain": None,
                        # "measure": "Clinical Diagnosis Demographics",
                        # "target_column": "DX_01",
                        # "transform": "factorize",
                        # "outname": "DX_01_factorize"
                        # },
                        # {"assessment": "Clinical Measures",
                        # "domain": None,
                        # "measure": "Children's Global Assessment Scale",
                        # "target_column": "CGAS,CGAS_Score",
                        # "transform": "numeric",
                        # "outname": "CGAS,CGAS_Score_numeric"
                        # }
                    ],
                },
                "preprocessing": {
                    "numeric": [
                        [
                            "sklearn.impute",
                            "SimpleImputer",
                            {
                                "strategy": "mean"
                            }
                        ],
                        [
                            "sklearn.preprocessing",
                            "StandardScaler",
                            {}
                        ]
                    ],
                # "category": [
                #     [
                #         "sklearn.impute", 
                #         "SimpleImputer", 
                #         {
                #             "strategy": "most_frequent"
                #         }
                #     ], [
                #         "sklearn.preprocessing", 
                #         "OneHotEncoder", 
                #         {"handle_unknown": "ignore", "sparse": False}
                #     ]
                # ]
            },
            "min_num_participants": 2000
            }
    outpath = os.path.join(out_dir, 'features-parent_spec.json')
    io.save_dict_as_JSON(outpath, spec_info)

    return outpath


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

    # transform the data
    df_transformed = preprocesser.fit_transform(dataframe_final)

    # get transformed feature names (on fitted transformers only)
    feature_names =  get_feature_names(column_transformer=preprocesser)

    # make pandas dataframe from transformed data
    df_transformed = pd.DataFrame(df_transformed, columns=feature_names)

    # add `col_to_ignore` back in
    if cols_to_ignore is not None:
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
    # c%dheck_is_fitted(column_transformer)
    
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
    
    # loop over pipelines
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
            try:
                feature_names.extend(get_names(trans))
            except:
                pass

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
        domains = info['Domain'].unique().tolist()
    elif 'Measure' in info.columns:
        domains = info['Measure'].unique().tolist()

    return {assessment: domains + ['all']}


def get_measures(assessment='Child Measures', domain='Cognitive Testing'):
    """get measures for `assessment` and `domain`. See `Assessment_List_2019.xlsx` for `assessment` and `domain`

    Args:
        assessment (str): options: 'Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'
        domain (str): specific for each assessment. if 'all', then measures for all domains are returned
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

    # return measures if both domain and measures are present
    if sum(info.columns.isin(['Domain', 'Measure']))==2:
        if domain is not 'all':
            measures = info[info['Domain']==domain]['Measure'].tolist()
        else:
            measures = []
            for name, group in info.groupby('Domain'):
                measures.extend(group['Measure'].tolist())
    else:
        measures = info['Measure']

    # add an exception here if the domain is `Interview_of_Emotional_and_Psychological_Function`
    # then additional parsed intake interview measures need to be added
    if all((assessment=='Parent Measures', domain=='Interview of Emotional and Psychological Function')):
        try:
            measures.extend(['Intake Interview PreInt Demos Fam',
                            'Intake Interview PreInt DevHx',
                            'Intake Interview PreInt EduHx',
                            'Intake Interview PreInt FamHx',
                            'Intake Interview PreInt FamHx RDC',
                            'Intake Interview PreInt Lang',
                            'Intake Interview PreInt TxHx'])
        except:
            pass

    return {domain: measures}


def _get_feature_combinations(parent_spec):
    """gets combinations of assessment*domain*measure to make feature files from `parent_spec`

    horrible code -- need to rewrite

    Args:
        parent_spec (str): full path to master spec file. saved in `FEATURE_DIR`
    """
    from hbn import io

    parent_spec = io.read_json(parent_spec)

    spec_info = []
    for assess in parent_spec['data']['features']['assessment']:
        domains = get_domains(assess)[assess]
        domains.remove('all')
        for domain in domains:
            measures = get_measures(assess, domain)[domain]
            for measure in measures:
                spec_info.append({'assessment': assess,
                        'domains': domain,
                        'measures': measure,
                        })
    return spec_info


def _setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
