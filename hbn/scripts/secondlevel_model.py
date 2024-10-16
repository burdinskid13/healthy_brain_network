import warnings
warnings.filterwarnings("ignore")
import os
import shutil
import pandas as pd
import glob
import click
from pathlib import Path

from hbn import io

def _get_best_classifier(firstlevel, metric='roc_auc_score'):
    """get best model classifier from `firstlevel_model`

    Args:
        firstlevel_model (str): fullpath to firstlevel model dir
        metric (str): name of metric to use to select best classifier. Default is 'fl_score'
    Returns:
        best_classifier (str): name of best classifier
    """
    # load model summary

    df = pd.read_csv(os.path.join(firstlevel, 'model-summary.csv'), engine='python')

    # groupby classifier
    group_clf = df.query('data=="model-data"'
                            ).groupby('clf'
                                ).apply(lambda x: x[metric].mean()
                                    ).reset_index(name='score'
                                        ).sort_values(by='score', ascending=False)

    # get best classifier
    best_classifier = group_clf['clf'].iloc[0]

    return best_classifier


def _get_specs(
        clf, 
        model_spec, 
        feature_importances, 
        feat=10, 
        which_features='top',
        ):
    """ get model spec info

    Args:
        clf (str): name of best classifier
        model_spec (str): fullpath to model spec
        model_features (str): fullpath to model features
        feature_importances (str): fullpath to feature importances
        feat (int): number of features to keep
        which_features (str): which features to keep
        feature_strategy (str): intersection or union
    Returns:
        info (dict): model spec info
    """
    
    # load files
    df_feat = pd.read_csv(feature_importances, engine='python')

    # load model spec info
    info = io.load_json(model_spec)
    info_filter = info.copy()

    # update pydraml spec
    info_filter['feature_threshold'] = f'{which_features}_{feat}_selected_features'

    for clf_info in info['clf_info']:

        if clf in clf_info[-1][1]:

            # index by classifier
            df_clf = df_feat[df_feat['clf'] == clf]

            # get features
            if feat is not None:
                df_clf = df_clf.sort_values(by='feature_importances', ascending=False)
                if which_features=='top':
                    filtered_features = df_clf['feature_importances_names'].head(feat).tolist()
                elif which_features=='bottom':
                    filtered_features = df_clf['feature_importances_names'].tail(feat).tolist()
                elif which_features=='all-minus-top':
                    filtered_features = df_clf.loc[feat:, 'feature_importances_names'].tolist()
            else:
                filtered_features = df_clf['feature_importances_names'].tolist()

            # assign new features to spec file
            info_filter['x_indices'] = filtered_features

            # update classifier
            info_filter['clf_info'] = [clf_info] 

            # update feature selection (should never be doing within CV feature selection in secondlevel modeling)
            info_filter['feature_selection'] = False
            info_filter['feature_selection_strategy'] = None     
    
    return info_filter


def make_model_spec(clf, firstlevel, secondlevel, splits, feat=10, which_features='top'):
    """ make secondlevel model spec

    Args:
        clf (str): name of best classifier
        firstlevel_model_dir (str): fullpath to firstlevel model dir
        secondlevel_model_dir (str): fullpath to secondlevel model dir
        splits (list of str): list of splits
    """

    # check if secondlevel dir exists
    io.make_dirs(secondlevel)

    # get features
    feature_importances = f'{firstlevel}/feature_importance.csv'

    # make model spec for secondlevel features
    for data in splits:

        # get specs
        spec = f'{firstlevel}/model_spec-{data}.json'
        spec_info = _get_specs(clf, spec, feature_importances, feat, which_features) 

        # copy model features to secondlevel directory
        shutil.copy(f'{firstlevel}/features-{data}.csv', secondlevel)

        # copy participant index to secondlevel directory
        shutil.copy(f'{firstlevel}/participant_index.csv', secondlevel)

        # save out secondlevel model specs to disk
        io.save_json(fpath=os.path.join(secondlevel, f'model_spec-{data}.json'), dict=spec_info)
        print(f'created secondlevel model spec file: model_spec-{data}.json and saved to {secondlevel}')


def run_model(features, model_spec, model_dir, cache_dir=None):
    """ run model train and model summary

    Args:
        features (str): full path to features file
        model_spec (str): full path to model spec or dict
        model_dir (str): full path to model directory (where `model_spec` and `features` are).
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    from hbn.scripts import make_model_summary
    from hbn.models import train_model

    print(f'model dir is :{model_dir}', flush=True)

    # define directory where model results will be saved
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # first level - run model
    train_model.train(
                    model_spec=model_spec,
                    features=features,
                    out_dir=model_dir,
                    cache_dir=cache_dir
                    )

    # get results file
    results = glob.glob(f'{model_dir}/*out*/*results*.pkl')[0] # should just be one file

    # second level - make summary
    make_model_summary.run(
                    results, # fullpath to results (.pkl)
                    model_spec,
                    out_dir=model_dir,
                    methods=['feature'] # feature interpretability based on feature or permuation importances
                    )


def run(firstlevel, secondlevel, cache_dir=None, feat=10, which_features='top'):
    """ run secondlevel model

    Args:
        firslevel_dir (str): full path to firstlevel model directory
        secondlevel_dir (str): full path to secondlevel model directory
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # get best classifier from firstlevel (if there was more than one)
    clf = _get_best_classifier(firstlevel)

    # make secondlevel model (train and test) for best classifier
    fnames = glob.glob(os.path.join(firstlevel, '*model_spec*'))
    specs = [Path(s).name.replace('model_spec-', '').replace('.json', '') for s in fnames]

    # try and make model spec but if it fails, just move on
    make_model_spec(clf, 
                    firstlevel, 
                    secondlevel, 
                    splits=specs, 
                    feat=feat, 
                    which_features=which_features
                    )

    # run secondlevel model (train)
    run_model(
        features=f'{secondlevel}/features-train.csv',
        model_spec=f'{secondlevel}/model_spec-train.json',
        model_dir=secondlevel,
        cache_dir=cache_dir
        )


if __name__ == '__main__':
    run()