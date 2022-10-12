from math import perm
import os
import re
import glob
from stat import FILE_ATTRIBUTE_INTEGRITY_STREAM
from wsgiref.simple_server import demo_app
import click
from pathlib import Path

from hbn import io
from hbn.constants import Defaults

import warnings
warnings.filterwarnings("ignore")

def parse_phenotypic_data():
    """
    """
    from hbn.data import make_dataset

    # parse phenotypic data
    assessments = ['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures']
    for assessment in assessments:
        make_dataset.parse_phenotypic_data(assessment=assessment)


def make_feature_specs():
    """make feature sets (json spec files + feature csv files)
    """
    from hbn.features import build_features

    def _get_iterables():
        """clunky code - need to rewrite
        """
        # get assessments + domains
        assessments = ['Child Measures', 'Parent Measures', 'Teacher Measures']
        targets = ['DX_01_Cat_binarize'] # 'DX_01_Cat_factorize', 'Sex_binarize', 'CGAS_Score'
        spec_info = []
        for assess in assessments:
            domains = build_features.get_domains(assess)[assess]
            for target in targets:
                target_type = 'categorical'
                if target=='CGAS_Score':
                    target_type = 'numeric'
                for domain in domains:
                    measures = build_features.get_measures(assess, domain)[domain]
                    for measure in measures:
                        spec_info.append({'assessment': assess,
                                    'domains': domain,
                                    'measures': measure, # 'all'
                                    'target': target,
                                    'target_type': target_type
                                    })
        return spec_info

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
                    "min_num_participants": 2000
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


def make_model_specs():
    """make model specs (json spec files)
    """

    # grab feature specs and make model specs
    feature_dir = os.path.join(Defaults.BASE_DIR, "features")
    fpaths = glob.glob(os.path.join(feature_dir, '*.json'))

    # hardcode classifiers
    clfs = {'categorical': [
                    ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
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
                "n_splits": 50,
                "test_size": 0.2,
                "clf_info": clf,
                "permute": [True, False],
                "gen_feature_importance": True,
                "gen_permutation_importance": True,
                "permutation_importance_n_repeats": 5,
                "permutation_importance_scoring": "accuracy",
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


def run_model_pipeline_firstlevel(
    specs=None, 
    cachedir='/Users/maedbhking/pydra-ml/cache-wf/',
    include_pattern='*classifier*DX_01_Cat_binarize*json'):
    """ run predictive models using pydra-ml. must provide `spec_file` json and `filename` in `spec_file` must be a csv of features saved in ../features/

    Args:
        spec_file (str or None or list of str): default is None
        tmpdir (str): default is '/Users/maedbhking/pydra-ml/cache-wf/'
        include_pattern (str): include pattern for models to run
    Returns: 
        saves (pickled) model to ../data/interim/
    """
    # load libraries
    import pandas as pd
    import shutil
    from pydra_ml.classifier import gen_workflow, run_workflow

    # figure out spec files
    if specs is None:
        specs = glob.glob(os.path.join(Defaults.BASE_DIR, "models", include_pattern))
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

        print(f'running {spec_file}...\n')
        wf = gen_workflow(spec_info, cache_dir=cachedir)
        run_workflow(wf, "cf", {"n_procs": 1})

        # move model output to new directory + add model spec file
        out_dir = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
        shutil.copy(spec_fpath, out_dir[0])
        shutil.move(out_dir[0], Defaults.MODEL_DIR)


def run_model_pipeline_secondlevel():
    """Makes model and feature summary files from results output from `run_model_pipeline_firstlevel`

    Saves output in `../interim/models/`
    """
    from hbn.models import second_level_modeling as second_level

    def _save_to_existing_file(dataframe, fpath):
        import pandas as pd

        df = pd.DataFrame()
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
        df_out = pd.concat([df, dataframe])
        df_out.to_csv(fpath, index=False)

    # grab list of models
    model_dirs = glob.glob(os.path.join(Defaults.MODEL_DIR, '*out-localspec*/*results*.pkl*'))
    spec_dirs = glob.glob(os.path.join(Defaults.MODEL_DIR, '*out-localspec*/*.json*'))
    
    for (model_fpath, spec_fpath) in zip(model_dirs, spec_dirs):

        # get model name
        modelname = Path(model_fpath).stem.split('-')[1] # `results-<modelname>`
        clf = Path(spec_fpath).name.split('-')[0]

        # get outpaths
        model_outpath = os.path.join(Defaults.MODEL_DIR, f'{clf}-all-phenotypic-models-performance.csv')
        feat_outpath = os.path.join(Defaults.MODEL_DIR, f'{clf}-feature_importance.csv')
        perm_outpath = os.path.join(Defaults.MODEL_DIR, f'{clf}-permutation_importance.csv')

        # load results
        results = second_level.load_results(fpath=model_fpath)

        # make feature summary (and save to disk)
        feature_df, permutation_df = second_level.get_features(
            results=results, 
            model=modelname,
            feature_importance=True, 
            permutation_importance=False
            )

        # get model summary (and save to disk)
        model_dataframe = second_level.get_model_summary(
                        results=results, 
                        spec_file=spec_fpath, 
                        )
        model_dataframe['model'] =  modelname

        # save to disk
        _save_to_existing_file(dataframe=feature_df, fpath=feat_outpath)
        _save_to_existing_file(dataframe=permutation_df, fpath=perm_outpath)
        _save_to_existing_file(dataframe=model_dataframe, fpath=model_outpath)


@click.command()
@click.option("--cachedir")
@click.option("--parse-data/--no-parse-data", default=False)
@click.option("--feature-specs/--no-feature-specs", default=True)
@click.option("--model-specs/--no-model-specs", default=True)
@click.option("--run-models-first/--no-run-models-first", default=True)
@click.option("--run-models-second/--no-run-models-second", default=True)

def run(
    cachedir='/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/',
    parse_data=False,
    feature_specs=False,
    model_specs=False,
    run_models_first=True,
    run_models_second=True,
    ):
    """ Entire processing workflow for processing phenotypic data from parsing data to running predictive models

    Args: 
        cachedir (str): full path to model cache directory.
        parse_data (bool): parse data from `/nese/mit/group/sig/projects/hbn/phenotype/data-2022-08-24T16_37_18.263Z.csv`. default is False because data have already been parsed and saved on OpenMind.
        feature_specs (bool): default is True. Saves feature specs (json and csv files) to `/om2/user/maedbh/healthy_brain_network/features`
        model_specs (bool): default is True. Saves model specs (json files) to `/om2/user/maedbh/healthy_brain_network/models`
        run_models_first (bool): default is True. Runs main predictive modeling routine: uses `https://github.com/nipype/pydra-ml` 
        run_models_second (bool): default is True. Wrapper function applied to output from `pydra-ml` to create model summaries, which are saved in `/nese/mit/group/sig/projects/hbn/phenotype/interim/models` 
            on openmind I use: '/home/maedbh/.cache/pydra-ml/cache-wf/
            on savio I use '/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/'
            on local I use '/Users/maedbhking/pydra-ml/cache-wf/'
    """
    # make cachedir if it doesn't exist
    io.make_dirs(cachedir)

    # FIRST STEP
    if parse_data:
        parse_phenotypic_data()

    # SECOND STEP
    if feature_specs:
        make_feature_specs()

    # THIRD STEP
    if model_specs:
        make_model_specs()

    # RUNNING MODELS (FIRST LEVEL)
    if run_models_first:
        run_model_pipeline_firstlevel(cachedir=cachedir)
    
    # RUNNING MODELS (SECOND LEVEL)
    if run_models_second:
        run_model_pipeline_secondlevel()

if __name__ == "__main__":
    run()


    