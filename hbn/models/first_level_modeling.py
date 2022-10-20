

def make_specs():
    """make model specs (json spec files)
    """
    import os
    import glob
    from hbn.constants import Defaults
    from hbn import io
    from pathlib import Path

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


def run_pipeline(
    spec_file, 
    features,
    cachedir='/Users/maedbhking/pydra-ml/cache-wf/'):
    """ run predictive models using pydra-ml. must provide `spec_file` json and `filename` in `spec_file` must be a csv of features saved in ../features/

    Args:
        spec_file (str): full path to model spec file
        features (str or pd dataframe): fullpath to features file or dataframe containing features
        cachedir (str): default is '/Users/maedbhking/pydra-ml/cache-wf/'
    Returns: 
        saves (pickled) model to ../data/interim/
    """
    # load libraries
    import os
    import pandas as pd
    import glob
    import shutil
    from pydra_ml.classifier import gen_workflow, run_workflow

    from hbn import io

    # load json
    spec_info = io.read_json(spec_file)

    # get features
    if isinstance(features, str):
        csv_file = os.path.join(Defaults.BASE_DIR, "features", spec_info['filename'])
        
    dataframe = pd.read_csv(csv_file)
    spec_info['filename'] = csv_file # full path to csv file

    spec_info['x_indices'] = range(1,len(dataframe.columns)-1)

    print(f'running {spec_file}...\n')
    wf = gen_workflow(spec_info, cache_dir=cachedir)
    run_workflow(wf, "cf", {"n_procs": 1})

    # move model output to new directory + add model spec file
    out_dir = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
    shutil.copy(spec_file, out_dir[0])
    shutil.move(out_dir[0], Defaults.MODEL_DIR)