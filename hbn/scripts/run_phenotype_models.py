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
        spec_dir (str): full path to folder where model specs are saved
    """
    from pathlib import Path
    import glob
    import os
    from hbn import io
    from hbn.models import predictive_modeling

    print('running first level')

    # get model specs
    specs = glob.glob(os.path.join(spec_dir, '*.json*'))

    if specs:
        # get model directory (where pydra-ml outputs are stored)
        ct_name = Path(spec_dir).name # same name as model spec dir
        out_dir = os.path.join(Defaults.MODEL_DIR, ct_name)

        # loop over models
        for model_spec in specs:

            # make new directory that is model specific
            spec_info = io.read_json(model_spec)
            dirn = spec_info['filename'].split("_")[-1].split(".")[0]
            model_dir = os.path.join(out_dir, f'model_{dirn}')

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
                    out_dir=out_dir
                    )


if __name__ == "__main__":
    run()


    