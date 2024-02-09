import os
import glob

import warnings
warnings.filterwarnings("ignore")


def run(features, model_spec, model_dir, cache_dir=None):
    """ run model train and model summary

    Args:
        features (str): full path to features file
        model_spec (str): full path to model spec
        model_dir (str): full path to model directory (where `model_spec` and `features` are).
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    from hbn.scripts import train_model, make_model_summary

    print(f'model dir is :{model_dir}', flush=True)

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


    