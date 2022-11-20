from hbn.constants import Defaults


def make_model_spec(
    feature_fpath,
    out_dir=Defaults.MODEL_SPEC_DIR
    ):
    """make model specs (json spec files) from the feature specs stored in `FEATURE_DIR`.
    model specs are saved out to `out_dir`
    Args:
        dataframe (pd dataframe): features dataframe to be input to modeling pipeline
        out_dir (str): full path to model spec output directory. default is `Defaults.MODEL_SPEC_DIR`
    Returns:
        full outpath to `model_spec` JSON
    """
    ### CONTINUE THIS TOMORROW ### 
    
    import os
    from hbn import io
    from pathlib import Path

    # define spec file
    spec_info = {
                "filename": feature_fpath,
                "x_indices": [],
                "target_vars": [target_info['outname']],
                "group_var": None,
                "n_splits": 50,
                "test_size": 0.2,
                "clf_info": [
                            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
                            ],
                "permute": [True, False],
                "gen_feature_importance": True,
                "gen_permutation_importance": True,
                "permutation_importance_n_repeats": 5,
                "permutation_importance_scoring": "accuracy",
                "gen_shap": False,
                "nsamples": "auto",
                "l1_reg": "aic",
                "plot_top_n_shap": 10,
                "metrics": ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
            }
    
    # write out model spec to disk ../model_specs/
    spec_name = 'classifier' + Path(feature_spec).name.replace('features', '').replace('-spec', '')
    io.save_dict_as_JSON(fpath=os.path.join(out_dir, spec_name), data_dict=spec_info)
    print(f'save model specs to file for {spec_name}')

    return os.path.join(out_dir, spec_name)


def make_model_features(
            target_spec,
            feature_spec,
            participant_files,
            out_dir=Defaults.FEATURE_DIR
            ):
    import os
    import pandas as pd
    from pathlib import Path
    from hbn import io
    import random
    from hbn.features import build_features

    # load from json file
    target_info = io.read_json(target_spec)
    feature_info = io.read_json(feature_spec)
    participants = pd.DataFrame()
    for fpath in participant_files:
        participants = pd.concat([participants, pd.read_csv(fpath)])

    # get features (X)
    features = build_features.get_features(
                assessment=feature_info['assessment'],
                domains=[feature_info['domains']],
                measures=[feature_info['measures']],
                min_num_participants=feature_info['min_num_participants']
                )

    # preprocess
    features_processed =build_features.preprocess(
                    dataframe=features,   
                    clf_info=feature_info['preprocessing'],
                    cols_to_ignore=['Identifiers']
                    )

    # combine features, targets, participants into one dataframe
    features_participants = features_processed.merge(participants, on='Identifiers')
    identifiers = features_participants['Identifiers'].tolist()
    targets = build_features.get_targets(target_info=target_info, participants=identifiers)

    features_final = features_participants.merge(targets, on='Identifiers').drop(['Identifiers'], axis=1)

    #fname = Path(feature_spec).stem.replace('-spec', '') + '-' + target_info['outname'] + '.csv'
    random_number = round(random.random()*1000000000)
    outpath = os.path.join(out_dir, f'model_features_{random_number}.csv')
    features_final.to_csv(outpath, index=False)

    return outpath


def run_pipeline(
    model_spec, 
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/',
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
    import random
    from pydra_ml.classifier import gen_workflow, run_workflow

    # create cachedir if it hasn't already been created
    io.make_dirs(cachedir)

    # load model spec json
    spec_info = io.read_json(model_spec)
    
    # get final features for model
    dataframe = make_features(spec_info)

    random_number = round(random.random()*1000000000)
    dataframe.reset_index(drop=True).to_csv(os.path.join(cachedir, f'temporary_features_{random_number}.csv'), index=False)
    
    spec_info['filename'] = os.path.join(cachedir, f'temporary_features_{random_number}.csv') # full path to csv file
    spec_info['x_indices'] = range(1,len(dataframe.columns)-1)

    print(f'running {model_spec}...\n')
    print("spec info", spec_info)
    
    wf = gen_workflow(spec_info, cache_dir=cachedir)
    run_workflow(wf, "cf", {"n_procs": 1})

    # move model output to new directory + add model spec file
    out_dir = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
    shutil.copy(model_spec, out_dir[0])
    shutil.move(out_dir[0], out_dir)