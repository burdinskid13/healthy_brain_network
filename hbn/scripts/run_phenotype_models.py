import click
import warnings
warnings.filterwarnings("ignore")


@click.command()
@click.option("--cachedir")
@click.option("--first-level/--no-first-level", default=False)
@click.option("--second-level/--no-second-level", default=True)

def run(
    first_level=False,
    second_level=True,
    cachedir='/om2/user/maedbh/.cache/pydra-ml/cache-wf/',
    ):
    """run first level modeling pipeline

    Args: 
        model_spec (str): model spec filename (not full path)
        cachedir (str): full path to  cache directory for pydra-ml intermediary outputs
    """
    import glob
    import os
    import datetime
    from hbn.constants import Defaults
    from hbn.models import first_level_modeling as first_level
    from hbn.models import second_level_modeling as second_level

    specs = glob.glob(os.path.join(Defaults.MODEL_SPEC_DIR, '*classifier*'))

    if first_level:
        # loop over model specs
        ct = datetime.datetime.now()
        ct_name = '_'.join(f'{ct}'.split(' '))
        for model_spec in specs:
            first_level.run_pipeline(
                model_spec=model_spec, 
                spec_dir=Defaults.MODEL_SPEC_DIR, 
                out_dir=os.path.join(Defaults.MODEL_DIR, ct_name),
                cachedir=cachedir
                )

    # get models
    models = glob.glob(os.path.join(Defaults.MODEL_DIR, '*'))

    if second_level:
        for model_dir in models:
            results = glob.glob(os.path.join(model, '*out-localspec*'))
            # loop over results files
            for result in results:
                second_level.run_pipeline(
                    results_dir=result,
                    out_dir=model_dir
                    )

if __name__ == "__main__":
    run()


    