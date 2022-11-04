from hbn.constants import Defaults


def make_model_spec(
    feature_spec, 
    participants,
    out_dir=Defaults.MODEL_SPEC_DIR
    ):
    """make model specs (json spec files) from the feature specs stored in `FEATURE_DIR`.
    model specs are saved out to `out_dir`
    Args:
        feature_spec (str): full path to feature spec file
        participants (str): participant file name (should NOT be full path to filename)
        out_dir (str): full path to model spec output directory. default is `Defaults.MODEL_SPEC_DIR`
    Returns:
        full outpath to `model_spec` JSON
    """
    import os
    from hbn import io
    from pathlib import Path

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

    # load from json file
    feature_info = io.read_json(feature_spec)

    # get target type, anything other than 'numeric' is considered 'categorical' for modeling purposes
    # 'numeric' = regression; 'categorical' = 'classifier'
    target_type = feature_info['target_y']['transform']
    if target_type!='numeric':
        target_type = 'categorical'
        model = 'classifier'
    else:
         model = 'regression'

    # get classifier
    clf = clfs[target_type]

    # get metrics
    metric = metrics[target_type]

    # define spec file
    spec_info = {
            "filename": feature_info['filename'], 
            "participants": participants,
            "x_indices": [],
            "target_vars": [feature_info['target_y']['outname']],
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
    
    # write out model spec to disk ../model_specs/
    spec_name = model + Path(feature_spec).name.replace('features', '').replace('-spec', '')
    io.save_dict_as_JSON(fpath=os.path.join(out_dir, spec_name), data_dict=spec_info)
    print(f'save model specs to file for {spec_name}')

    return os.path.join(out_dir, spec_name)


def run_pipeline(
    model_spec, 
    features,
    participants,
    cachedir='/Users/maedbhking/pydra-ml/cache-wf/',
    out_dir=''):
    """ run predictive models using pydra-ml. must provide `model_spec` json and `filename` in `model_spec` must be a csv of features saved in ../features/

    Args:
        model_spec (str): full path to model spec file
        features (str or pd dataframe): fullpath to features file or dataframe containing features
        participants (str or pd dataframe): fullpath to participants file or dataframe containing column 'Identifiers' to indicate participants
        cachedir (str): default is '/Users/maedbhking/pydra-ml/cache-wf/'
        out_dir (str): full path to model output directory
    Returns: 
        saves (pickled) model to ../data/interim/
    """
    # load libraries
    import os
    import pandas as pd
    import glob
    import shutil
    from hbn import io
    from pydra_ml.classifier import gen_workflow, run_workflow

    from hbn import io

    # create cachedir if it hasn't already been created
    io.make_dirs(cachedir)

    # load json
    spec_info = io.read_json(model_spec)
    
    # get features
    if isinstance(features, str):
        features = os.path.join(features)

    # get participants
    if isinstance(participants, str):
        participants = os.path.join(participants)

    # load dataframes for features and participants
    df_features = pd.read_csv(features)
    df_participants = pd.read_csv(participants)

    # make new feature file, merging on common 'Identifiers'
    # save out file temporarily
    features_final = df_features.merge(df_participants, on='Identifiers').drop(columns=['Identifiers'])
    features_final.reset_index(drop=True).to_csv(os.path.join(cachedir, f'temporary-features.csv'), index=False)

    spec_info['filename'] = os.path.join(cachedir, f'temporary-features.csv') # full path to csv file
    spec_info['x_indices'] = range(1,len(features_final.columns)-1)

    print(f'running {model_spec}...\n')
    wf = gen_workflow(spec_info, cache_dir=cachedir)
    run_workflow(wf, "cf", {"n_procs": 1})

    # move model output to new directory + add model spec file
    out_dir = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
    shutil.copy(model_spec, out_dir[0])
    shutil.move(out_dir[0], out_dir)