import warnings
warnings.filterwarnings("ignore")
import os
import shutil
import pandas as pd
import click

from hbn import io
from hbn.scripts import run_model

def _get_best_classifier(firstlevel_model, metric='f1_score'):
    """get best model classifier from `firstlevel_model`

    Args:
        firstlevel_model (str): fullpath to firstlevel model dir
        metric (str): name of metric to use to select best classifier. Default is 'fl_score'
    Returns:
        best_classifier (str): name of best classifier
    """
    # load model summary

    df = pd.read_csv(os.path.join(firstlevel_model, 'model-summary.csv'), engine='python')

    # groupby classifier
    group_clf = df.query('data=="model-data"'
                            ).groupby('clf'
                                ).apply(lambda x: x[metric].mean()
                                    ).reset_index(name='score'
                                        ).sort_values(by='score', ascending=False)

    # get best classifier
    best_classifier = group_clf['clf'].head(1)[0]

    return best_classifier


def _get_specs(clf, model_spec, model_features, feature_importances):
    """ get model spec info

    Args:
        clf (str): name of best classifier
        model_spec (str): fullpath to model spec
        model_features (str): fullpath to model features
        feature_importances (str): fullpath to feature importances
    Returns:
        info (dict): model spec info
    """
    
    # load files
    model_features = pd.read_csv(model_features, engine='python')
    df_feat = pd.read_csv(feature_importances, engine='python')

    # load model spec info
    info = io.load_json(model_spec)

    for clf_info in info['clf_info']:

        if clf in clf_info[-1][1]:

            # index by classifier
            df_clf = df_feat[df_feat['clf'] == clf]

            # get top features
            top_features =  df_clf[df_clf['top_features']==True]['feature_names'].tolist()

            # assign new features to spec file
            info['x_indices'] = top_features

            # update classifier
            info['clf_info'] = [clf_info]

        else:
            info = {}
    
    return info


def make_model_spec(clf, firstlevel_model_dir, secondlevel_model_dir, splits):
    """ make secondlevel model spec

    Args:
        clf (str): name of best classifier
        firstlevel_model_dir (str): fullpath to firstlevel model dir
        secondlevel_model_dir (str): fullpath to secondlevel model dir
        splits (list of str): list of splits
    """

    # check if secondlevel dir exists
    io.make_dirs(secondlevel_model_dir)

    # get features
    feature_importances = f'{firstlevel_model_dir}/feature_importance.csv'

    # make model spec for secondlevel features
    for data in splits:

        # get spec and features
        spec = f'{firstlevel_model_dir}/model_spec-{data}.json'
        features = f'{firstlevel_model_dir}/features-{data}.csv'

        # get specs
        spec_info = _get_specs(clf, model_spec=spec, model_features=features, feature_importances=feature_importances)  
        spec_name = f'model_spec-{data}.json'

        # copy model features to secondlevel directory
        shutil.copy(features, secondlevel_model_dir)

        # copy participant index to secondlevel directory
        fpath = os.path.join(firstlevel_model_dir, 'participant_index.csv')
        shutil.copy(fpath, secondlevel_model_dir)

        # save out secondlevel model specs to disk
        outpath = os.path.join(secondlevel_model_dir, spec_name)
        io.save_json(fpath=outpath, dict=spec_info)
        print(f'created secondlevel model spec file: {spec_name} and saved to {secondlevel_model_dir}')

@click.command()
@click.option("--model_dir", required=True)
@click.option("--cache_dir", required=False)
def run(model_dir, cache_dir=None):
    """ run secondlevel model

    Args:
        model_dir (str): full path to parent model directory
        train (str): name of train split
        test (str): name of test split
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # define directories
    firstlevel_model_dir = model_dir
    secondlevel_model_dir = firstlevel_model_dir + '_secondlevel'

    # get best classifier from firstlevel (if there was more than one)
    clf = _get_best_classifier(firstlevel_model_dir)

    # make secondlevel model (train and test)
    make_model_spec(clf, firstlevel_model_dir, secondlevel_model_dir, splits=['train', 'test'])

    # run secondlevel model (train)
    features = f'{secondlevel_model_dir}/features-train.csv'
    spec = f'{secondlevel_model_dir}/model_spec-train.json'

    run_model.run(
        features=features,
        model_spec=spec,
        model_dir=secondlevel_model_dir,
        cache_dir=cache_dir
        )


if __name__ == '__main__':
    run()