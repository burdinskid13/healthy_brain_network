import warnings
warnings.filterwarnings("ignore")
import os
import ast
import random
import shutil
import glob
import pandas as pd

from src import io
from src.features import build_features

def make_secondlevel_spec(model_spec, model_features, feature_importances):
    """make secondlevel model using firstlevel model

    Args:
        model_spec (str): fullpath to model spec
        model_features (str): fullpath to model features
        feature_importances (str): fullpath to feature importances
    Returns:
        model_spec (dict): secondlevel model spec
    """

    # load files
    model_features = pd.read_csv(model_features)
    feature_importances = pd.read_csv(feature_importances)

    # loop over classifiers (if there are more than one)
    classifiers = feature_importances['clf'].unique()
    model_spec_all = []; model_spec_names = []
    for idx, clf in enumerate(classifiers):

        # load model spec info
        model_spec_info = io.load(model_spec)

        # index by classifier
        df_clf = feature_importances[feature_importances['clf'] == clf]

        # get top features
        top_features = df_clf[df_clf['features']==True]['feature_names'].tolist()

        # get indices of top features
        indices = []
        for feat in top_features:
            indices.append(model_features.columns.get_loc(feat))
        
        # assign new features to spec file
        model_spec_info['x_indices'] = indices

        # update classifier
        model_spec_info['clf_info'] = [model_spec_info['clf_info'][idx]]

        model_spec_all.append(model_spec_info)
        model_spec_names.append(f'model-spec-{clf}.json')
    
    return model_spec_all, model_spec_names

def run(firstlevel_model_dir, secondlevel_model_dir):
    """ Makes secondlevel model based on firstlevel model

    Args:
        firstlevel_model_dir (str): fullpath to firstlevel model directory
        secondlevel_model_dir (str): fullpath to secondlevel model directory

    Returns:
        saves out secondlevel models to disk
    """

    # check if secondlevel model directory exists
    io.make_dirs(secondlevel_model_dir)

    # get results from model dir
    results = glob.glob(f'{firstlevel_model_dir}/*out*/*results*.pkl')[0] # should just be one file

    # load specs and features and feature importances from `firstlevel_model_dir`
    model_spec = glob.glob(f'{firstlevel_model_dir}/*model_spec*')[0]
    model_features = glob.glob(f'{firstlevel_model_dir}/*features*')[0]
    feature_importances = glob.glob(f'{firstlevel_model_dir}/*feature_importances*')[0]

    # make model spec for secondlevel features
    secondlevel_specs, spec_names = make_secondlevel_spec(model_spec, model_features, feature_importances)

    # copy model features to secondlevel directory 
    shutil.copy(model_features, secondlevel_model_dir)

    # save out secondlevel model specs to disk
    for info, name in zip(secondlevel_specs, spec_names):
        io.save_json(os.path.join(secondlevel_model_dir, name), dict=info)
        print(f'created secondlevel model spec: {name} and saved to {secondlevel_model_dir}')

if __name__ == 'main':
    run()


