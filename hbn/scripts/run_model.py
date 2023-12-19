from hbn.constants import Defaults
import os
import click
import glob

import warnings
warnings.filterwarnings("ignore")

@click.command()
@click.option("--model_dir", required=True)
@click.option("--cache_dir", required=True)

def run(model_dir, cache_dir=None):
    """ run model train and model summary

    Args:
        model_dir (str): full path to model directory (where `model_spec` and `features` are).
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    from hbn.scripts import train_model, make_model_summary

    print(f'model dir is :{model_dir}')

    # fullpath to features
    features = glob.glob(f'{model_dir}/*features*')[0] # should only be one feature file

    # fullpath to model spec
    model_spec = glob.glob(f'{model_dir}/*model_spec*')[0] # should only be one spec file

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
    make_model_summary.run(
                    results, # fullpath to results (.pkl)
                    model_spec,
                    out_dir=model_dir,
                    methods=['feature'] # feature interpretability based on feature or permuation importances
                    )


if __name__ == "__main__":
    run()


    