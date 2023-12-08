import warnings
warnings.filterwarnings("ignore")

def run(
    model_spec,
    features,
    out_dir,
    cache_dir=None
    ):
    """run predictive models using pydra-ml. must provide `model_spec` and `features`.

    Args:
        model_spec (str): full path to model spec file.
        features (str): full path to features file.
        out_dir (str): directory where model results will be saved. Default is home directory
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    Returns:
        saves pickled model to `out_dir`
    """
    # load libraries
    import os
    from src import io
    from pathlib import Path
    from pydra_ml.classifier import gen_workflow, run_workflow

    # make cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # make out_dir
    if out_dir is None:
        out_dir = os.path.expanduser('~')

    # create `cache dir` and `out_dir` if they don't exist
    io.make_dirs(cache_dir)
    io.make_dirs(out_dir)

    # load model spec
    spec_info = io.load_json(model_spec)

    print(f'training model {model spec}', flush=True)

    # assign fullpath to features csv
    spec_info['filename'] = features

    # change directory to `out_dir`
    os.chdir(out_dir)
    print(f'changing directory to {out_dir}', flush=True)

    # run workflow
    wf = gen_workflow(spec_info, cache_dir=cache_dir)
    run_workflow(wf, "cf", {'n_procs': 1})

if __name__ == '__main__':
    run()