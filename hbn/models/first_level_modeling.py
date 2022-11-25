from hbn.constants import Defaults


def make_model_spec(
            filename,
            target_spec,
            feature_spec,
            participants,
            out_dir=Defaults.MODEL_SPEC_DIR
             ):
    """make model specs (json spec files) from the feature specs stored in `FEATURE_DIR`.
    model specs are saved out to `out_dir`
    Args:
        filename (str): full path to features filename. Saved in MODEL_SPEC_DIR
        target_spec (str): full path to target spec file. SAVED IN FEATURE_DIR
        feature_spec (str): full path to feature spec file. SAVED IN FEATURE_DIR
       participants (list of str): list of fullpaths to participant files. Example ['../train_participants-ADHD.csv', '../train_participants-No_Diagnosis_Given.csv']
        out_dir (str): full path to model spec output directory. default is `Defaults.MODEL_SPEC_DIR`
    Returns:
        full outpath to `model_spec` JSON
    """
    import re
    import os
    from hbn import io
    from pathlib import Path

    target_info = io.read_json(target_spec)
    feature_info = io.read_json(feature_spec)

    # get participant filenames
    participant_fnames = []
    for participant in participants:
        participant_fnames.append(Path(participant).name)

    # define spec file
    spec_info = {
                "filename": Path(filename).name,
                "feature_spec": feature_info,
                "target_spec": target_info,
                "participants": participant_fnames,
                "x_indices":[],
                "target_vars": [target_info['outname']],
                "group_var": None,
                "n_splits": 50,
                "test_size": 0.2,
                "clf_info": [
                            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
                            ],
                "permute": [True, False],
                "gen_feature_importance": True,
                "gen_permutation_importance": False,
                "permutation_importance_n_repeats": 5,
                "permutation_importance_scoring": "accuracy",
                "gen_shap": False,
                "nsamples": "auto",
                "l1_reg": "aic",
                "plot_top_n_shap": 10,
                "metrics": ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
            }
    
    # write out model spec to disk ../model_specs/
    spec_name = 'classifier-' + '_'.join(re.split(r'_|,|/| ', feature_info['measures'])) + '-' + target_info['outname'] + '-spec.json'
    io.save_dict_as_JSON(fpath=os.path.join(out_dir, spec_name), data_dict=spec_info)
    print(f'save model specs to file for {spec_name}')

    return os.path.join(out_dir, spec_name)


def make_model_features(
            target_spec,
            feature_spec,
            participants
            ):
    """make model features

    Args:
        target_spec (str): full path to target spec file 
        feature_spec (str): full path to feature spec file
        participants (list of str): list of fullpaths to participant files. Example ['../train_participants-ADHD.csv', '../train_participants-No_Diagnosis_Given.csv']
    Returns: 
        features_final (pd dataframe): features to be input to modeling routine
    """
    import pandas as pd
    from hbn import io
    from hbn.features import build_features

    # load from json file
    target_info = io.read_json(target_spec)
    feature_info = io.read_json(feature_spec)

    # make participants dataframe
    participants_df = pd.DataFrame()
    for participant in participants:
        participants_df = pd.concat([participants_df, pd.read_csv(participant)])

    # get features (X)
    features = build_features.get_features(
                assessment=feature_info['assessment'],
                domains=[feature_info['domains']],
                measures=[feature_info['measures']],
                min_num_participants=feature_info['min_num_participants']
                )

    # preprocess
    features_processed = build_features.preprocess(
                    dataframe=features,   
                    clf_info=feature_info['preprocessing'],
                    cols_to_ignore=['Identifiers']
                    )

    # combine features, targets, participants into one dataframe
    features_participants = features_processed.merge(participants_df, on='Identifiers')
    identifiers = features_participants['Identifiers'].tolist()
    targets = build_features.get_targets(
                            target_info=target_info, 
                            participants=identifiers
                            )

    features_final = features_participants.merge(targets, on='Identifiers').drop(['Identifiers'], axis=1)

    return features_final


def run_pipeline(
    model_spec, 
    spec_dir,
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/',
    out_dir=Defaults.MODEL_DIR):
    """ run predictive models using pydra-ml. must provide `model_spec` json and `filename` in `model_spec` must be a csv of features saved in ../features/

    Args:
        model_spec (str): full path to model spec file
        spec_dir (str): model spec directory (where `filename` in model_spec is stored)
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

    # create cachedir if it hasn't already been created
    io.make_dirs(cachedir)
    io.make_dirs(out_dir)

    # load model spec json
    spec_info = io.read_json(model_spec)

    # load dataframe
    dataframe = pd.read_csv(os.path.join(spec_dir, spec_info['filename']))
    spec_info['x_indices'] =  range(1,len(dataframe.columns)-1)

    print(f'running {model_spec}...\n')
    print("spec info", spec_info)
    
    filename = os.path.join(spec_dir, spec_info['filename'])
    spec_info['filename'] = filename # full path to csv file
    wf = gen_workflow(spec_info, cache_dir=cachedir)
    run_workflow(wf, "cf", {"n_procs": 1})

    # move model output to new directory + add model spec file
    out_models = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
    shutil.move(model_spec, out_models[0])
    shutil.move(filename, out_models[0])
    shutil.move(out_models[0], out_dir)