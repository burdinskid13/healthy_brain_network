import click
import warnings
warnings.filterwarnings("ignore")


@click.command()
@click.option("--cachedir")
@click.option("--first-level/--no-first-level", default=False)
@click.option("--second-level/--no-second-level", default=True)

def run(
    cachedir='/om2/user/maedbh/.cache/pydra-ml/cache-wf/',
    first_level=False,
    second_level=True,
    ):
    """run first level modeling pipeline

    Args: 
        model_spec (str): model spec filename (not full path)
        cachedir (str): full path to  cache directory for pydra-ml intermediary outputs
    """
    import glob
    import os
    import datetime
    from hbn import io
    import shutil
    from hbn.constants import Defaults
    from hbn.models import first_level_modeling as first
    from hbn.models import second_level_modeling as second

    if first_level:
        print('running first level')
        specs = glob.glob(os.path.join(Defaults.MODEL_SPEC_DIR, '*classifier*'))
        # loop over model specs
        ct = datetime.datetime.now()
        ct_name = '_'.join(f'{ct}'.split(' '))
        for model_spec in specs:
            first.run_pipeline(
                model_spec=model_spec, 
                spec_dir=Defaults.MODEL_SPEC_DIR, 
                out_dir=os.path.join(Defaults.MODEL_DIR, ct_name),
                cachedir=cachedir
                )

    if second_level:
        # temporary file
        models = glob.glob(os.path.join(Defaults.MODEL_DIR, '*'))
        for model_dir in models:
            results = glob.glob(os.path.join(model_dir, '*out-localspec*'))
            for result in results:
                spec_file = glob.glob(os.path.join(result, '*.json'))[0]
                filename = io.read_json(spec_file)['filename']
                source = os.path.join(Defaults.MODEL_SPEC_DIR, filename)
                shutil.move(source, result)


if __name__ == "__main__":
    run()


    