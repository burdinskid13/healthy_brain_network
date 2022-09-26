import os
import re
import glob
from pathlib import Path

import pandas as pd
import shutil
from pydra_ml.classifier import gen_workflow, run_workflow

from hbn.data import make_dataset
from hbn.features import build_features
from hbn import io
from hbn.constants import Defaults

def parse_phenotypic_data():
    """
    """
    # parse phenotypic data
    assessments = ['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures']
    for assessment in assessments:
        make_dataset.parse_phenotypic_data(assessment=assessment)

def make_feature_specs():
    """make feature sets (json spec files + feature csv files)
    """

    for data in _get_iterables():

        # define spec filename
        vals = []
        for val in ['features', 'assessment', 'domains', 'measures', 'target']:
            if val in data.keys():
                vals.append('_'.join(re.split(r'_|,|/| ', data[val])))
            else:
                vals.append(val)
        spec_file = '-'.join(vals)

        # set spec + feature filenames
        spec_fpath = os.path.join(Defaults.BASE_DIR, "features", spec_file + '-spec.json')
        feature_fpath = os.path.join(Defaults.BASE_DIR, "features", spec_file + '.csv')

        # define spec file
        spec_info = {"filename": spec_file + '.csv',
                    "assessment": data['assessment'],
                    "domains": data['domains'],
                    "measures": data['measures'],
                    "target": data['target'],
                    "target_type": data['target_type'],
                    "preprocessing": {"numeric": [["sklearn.impute", "SimpleImputer", {"strategy": "mean"}], ["sklearn.preprocessing", "StandardScaler", {}]]},
                    "min_num_participants": 4000
                    }

        # get features (X and y) - make csv file
        df = build_features.get_data(
                            assessment=spec_info['assessment'],
                            domains=spec_info['domains'],
                            measures=spec_info['measures'],
                            target=spec_info['target'],
                            min_num_participants=spec_info['min_num_participants']
                            )

        df_processed = build_features.preprocess(
                            dataframe=df,   
                            clf_info=spec_info['preprocessing'],
                            cols_to_ignore=spec_info['target']
                            )

        # save json + csv to ../features/ only if there are X features (not just y target)
        if len(df_processed.columns)>1:
            io.save_dict_as_JSON(fpath=spec_fpath, data_dict=spec_info)
            df_processed.to_csv(feature_fpath, index=False)
            print(f'spec file and features saved to disk for {spec_file}')

def _get_iterables():
    """clunky code - need to rewrite
    """
    # get assessments + domains
    assessments = ['Child Measures', 'Parent Measures', 'Teacher Measures']
    targets = ['DX_01_Cat_binarize', 'Sex_binarize', 'CGAS_Score'] # 'DX_01_Cat_factorize',
    spec_info = []
    for assess in assessments:
        domains = build_features.get_domains(assess)[assess]
        for target in targets:
            target_type = 'categorical'
            if target=='CGAS_Score':
                target_type = 'numeric'
            for domain in domains:
                spec_info.append({'assessment': assess,
                            'domains': domain,
                            'measures': 'all',
                            'target': target,
                            'target_type': target_type
                            })
    return spec_info

def make_model_specs():
    """make model specs (json spec files)
    """

    # grab feature specs and make model specs
    feature_dir = os.path.join(Defaults.BASE_DIR, "features")
    fpaths = glob.glob(os.path.join(feature_dir, '*.json'))

    # hardcode classifiers
    clfs = {'categorical': [
                    [["sklearn.impute", "SimpleImputer"],
                    ["sklearn.preprocessing", "StandardScaler"],
                    ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]],
                    ],
            'numeric': [
                ["sklearn.linear_model","RidgeCV",{"fit_intercept": False}],
                ]
            }

    metrics = {'categorical': 
                ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score'],
               'numeric': 
               ["explained_variance_score", "mean_squared_error", "mean_absolute_error"]
            }

    # loop over feature filenames
    for fpath in fpaths:

        # load from json file
        feature_spec = io.read_json(fpath)
        target_type = feature_spec['target_type']

        # get classifier
        clf = clfs[target_type]

        # get metrics
        metric = metrics[target_type]

        # define spec file
        spec_info = {
                "filename": feature_spec['filename'], 
                "x_indices": [],
                "target_vars": [feature_spec['target']],
                "group_var": None,
                "n_splits": 15,
                "test_size": 0.2,
                "clf_info": clf,
                "permute": [True, False],
                "gen_feature_importance": True,
                "gen_permutation_importance": False,
                "permutation_importance_n_repeats": 5,
                "permutation_importance_scoring": None,
                "gen_shap": False,
                "nsamples": "auto",
                "l1_reg": "aic",
                "plot_top_n_shap": 10,
                "metrics": metric
                }
        
        if target_type=='categorical':
            model = 'classifier'
        elif target_type=='numeric':
            model = 'regression'
        
        # write out model spec to disk ../models/
        spec_name = model + Path(fpath).name.replace('features', '').replace('-spec', '')
        io.save_dict_as_JSON(fpath=os.path.join(Defaults.BASE_DIR, "models", spec_name), data_dict=spec_info)
        print(f'save model specs to file for {spec_name}')

def run_models(
    specs=None, 
    tmpdir='/Users/maedbhking/pydra-ml/cache-wf/'):
    """ run predictive models using pydra-ml. must provide `spec_file` json and `filename` in `spec_file` must be a csv of features saved in ../features/

    Args:
        spec_file (str or None or list of str): default is None
        tmpdir (str): default is '/Users/maedbhking/pydra-ml/cache-wf/'
    Returns: 
        saves (pickled) model to ../data/interim/
    """
    # figure out spec files
    if specs is None:
        specs = glob.glob(os.path.join(Defaults.BASE_DIR, "models", '*json'))
    elif specs is str:
        specs = [specs]
    
    # loop over model specs
    for spec_file in specs:
        # load json
        spec_fpath = os.path.join(Defaults.BASE_DIR, "models", spec_file)
        spec_info = io.read_json(spec_fpath)

        # get features
        csv_file = os.path.join(Defaults.BASE_DIR, "features", spec_info['filename'])
        dataframe = pd.read_csv(csv_file)
        spec_info['filename'] = csv_file # full path to csv file

        spec_info['x_indices'] = range(1,len(dataframe.columns)-1)

        wf = gen_workflow(spec_info, cache_dir=tmpdir)
        results = run_workflow(wf, "cf", {"n_procs": 1})

        # move model output to new directory + add model spec file
        out_dir = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
        shutil.copy(spec_fpath, out_dir[0])
        shutil.move(out_dir[0], Defaults.MODEL_DIR)
        shutil.rmtree("messages")

def run():
    """ Entire processing workflow for processing phenotypic data from parsing data to running predictive models
    """

    # First Step
    parse_phenotypic_data()

    # Second Step
    make_feature_specs()

    # Third step
    make_model_specs()

    # running models
    run_models()

if __name__ == "__main__":
    run()


    