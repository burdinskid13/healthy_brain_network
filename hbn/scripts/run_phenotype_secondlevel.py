import warnings
import click
from hbn.constants import Defaults
warnings.filterwarnings("ignore")

@click.command()
@click.option('--filter', required=False)

def run(filter='03-07-23'):
    """run second level modeling pipeline
    """
    import glob
    import os
    from hbn.models import predictive_modeling

    print('running second level')

    if filter is None:
        model_dirs = glob.glob(os.path.join(Defaults.MODEL_DIR, '*'))
    else:
        model_dirs = glob.glob(os.path.join(Defaults.MODEL_DIR, filter))
    print(model_dirs, filter)
    
    # loop over models
    for model_dir in model_dirs:
        
        results_dir = glob.glob(os.path.join(model_dir, '*model_*', '*out-localspec*'))
        
        # loop over results files
        for result_dir in results_dir:
            predictive_modeling.secondlevel_summary(
                results_dir=result_dir,
                out_dir=model_dir
                )

if __name__ == "__main__":
    run()
