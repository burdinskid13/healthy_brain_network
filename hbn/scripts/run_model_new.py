from src.constants import Defaults
import os
import glob

import warnings
warnings.filterwarnings("ignore")

def model_train_summary(model_dir='../bhs_demos', cache_dir=None):
    """ train model and get model summary of results

    Args:
        model_dir (str): full path to model directory. The following files should be in `model_dir`: `model_spec-{model_name}.json`, `features-{model_spec}.json`.
        There should only be one model_spec file and one features file.
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    from src.scripts import train_model, model_summary

    # fullpath to features
    features = glob.glob(f'{model_dir}/*features*')[0]

    # fullpath to model spec
    model_spec = glob.glob(f'{model_dir}/*model_spec*')[0]

    # define directory where model results will be saved
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # first level - run model
    train_model.run(
                    model_spec=model_spec,
                    features=features,
                    out_dir=model_dir,
                    cache_dir=cache_dir
                    )

    # get results file
    results = glob.glob(f'{model_dir}/*out*/*results*.pkl')[0] # should just be one file

    # second level - make summary
    model_summary.run(
                    results, # fullpath to results (.pkl)
                    model_spec,
                    out_dir=model_dir,
                    methods=['feature'] # feature interpretability based on feature or permuation importances
                    )


def run(model_dir=Dirs.model_dir, models=[], cache_dir=None):
    """ run model train and model summary

    Args:
        model_dir (str): full path to parent model directory (`../processed/models/`)
        models (list of str): list of model names to run. Default is all models in `model_dir`
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """

    # grab all models if list of models is empty
    if len(models)==0:
        models = os.listdir(model_dir)
    
    # loop over models
    for model in models:
        # get fullpath to model directory
        # model spec file and model features should also be in `model_path`
        model_path = os.path.join(model_dir, model)

        # train model and get model summary
        model_train_summary(model_dir=model_path, cache_dir=cache_dir)


if __name__ == "__main__":
    run()


    