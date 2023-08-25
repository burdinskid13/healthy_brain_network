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
    import shutil
    from hbn import io
    from hbn.models import predictive_modeling

    print('running first level', flush=True)

    # get model specs
    specs = glob.glob(os.path.join(spec_dir, '*.json*'))

    if specs:

        # loop over models
        print('looping over models', flush=True)
        for (idx, model_spec) in enumerate(specs):

            # make new directory that is model specific
            spec_info = io.read_json(model_spec)
            dirn = 'model_' + spec_info['filename'].split("_")[-1].split(".")[0]
            if idx>0:
                dirn = dirn + f'.{idx}' # new model directory

            predictive_modeling.run_pydra_ml(
                model_spec=model_spec, 
                out_dir=os.path.join(spec_dir, dirn),
                cachedir=cachedir
                )

            print('running model summary', flush=True)
            results_list = glob.glob(os.path.join(spec_dir, dirn, '*out-localspec*', '*results*.pkl'))
            
            # loop over results files (should just be one outspec folder per model directory)
            for result in results_list:
                predictive_modeling.model_summary(
                    results=result,
                    spec=model_spec,
                    out_dir=spec_dir
                    )
            
            # move specs and features into model dir
            shutil.move(model_spec, os.path.join(spec_dir, dirn))
            shutil.move(os.path.join(spec_dir, spec_info['filename']), os.path.join(spec_dir, dirn))


if __name__ == "__main__":
    run()


    