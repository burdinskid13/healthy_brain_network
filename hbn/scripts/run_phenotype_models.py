import click
import warnings
from hbn.constants import Defaults
warnings.filterwarnings("ignore")


@click.command()
@click.option("--cachedir")
@click.option("--spec_dir", required=False)

def run(
    cachedir='/om2/user/maedbh/.cache/pydra-ml/cache-wf/',
    spec_dir=Defaults.MODEL_SPEC_DIR,
    ):
    """run first level modeling pipeline

    Args: 
        cachedir (str): full path to  cache directory for pydra-ml intermediary outputs
        spec_dir (str): full pat to `MODEL_SPEC_DIR`
    """
    import glob
    import os
    import datetime
    from hbn.models import predictive_modeling

    print('running first level')

    # get model specs
    specs = glob.glob(os.path.join(spec_dir, '*.json*'))

    # loop over model specs
    ct = datetime.datetime.now()
    ct_name = '_'.join(f'{ct}'.split(' '))

    # get model directory (where pydra-ml outputs are stored)
    model_dir = os.path.join(Defaults.MODEL_DIR, ct_name)
    for model_spec in specs:
        predictive_modeling.run_pydra_ml(
            model_spec=model_spec, 
            spec_dir=spec_dir, 
            out_dir=model_dir,
            cachedir=cachedir
            )

    print('running second level')
    results = glob.glob(os.path.join(model_dir, '*out-localspec*'))
    # loop over results files
    for result in results:
        predictive_modeling.secondlevel_summary(
            results_dir=result,
            out_dir=model_dir
            )


if __name__ == "__main__":
    run()


    