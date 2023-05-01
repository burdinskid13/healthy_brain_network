import click
import warnings
from hbn.constants import Defaults
#warnings.filterwarnings("ignore")


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
        spec_dir (str): full path to folder where model specs are saved
    """
    from pathlib import Path
    import glob
    import os
    from hbn import io
    from hbn.models import predictive_modeling

    print('running first level', flush=True)

    # get model specs
    specs = glob.glob(os.path.join(spec_dir, '*.json*'))

    if specs:

        # loop over models
        print('looping over models', flush=True)
        for model_spec in specs:

            # make new directory that is model specific
            spec_info = io.read_json(model_spec)
            dirn = spec_info['filename'].split("_")[-1].split(".")[0]

            predictive_modeling.run_pydra_ml(
                model_spec=model_spec, 
                out_dir=os.path.join(spec_dir, f'model_{dirn}'),
                cachedir=cachedir
                )

            print('running second level', flush=True)
            results = glob.glob(os.path.join(spec_dir, f'model_{dirn}', '*out-localspec*'))
            # loop over results files
            for result in results:
                predictive_modeling.secondlevel_summary(
                    results_dir=result,
                    out_dir=spec_dir
                    )


if __name__ == "__main__":
    run()


    