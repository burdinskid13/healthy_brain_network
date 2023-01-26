import warnings
from hbn.constants import Defaults
warnings.filterwarnings("ignore")

def run(filter='2023-01-25_22:11:05.460227'):
    """run second level modeling pipeline
    """
    import glob
    import os
    from hbn.models import predictive_modeling

    print('running second level')
    model_dirs = glob.glob(os.path.join(Defaults.MODEL_DIR, f'*{filter}*')) # '*2023*'
    for model_dir in model_dirs:
        results = glob.glob(os.path.join(model_dir, '*out-localspec*'))
        # loop over results files
        for result in results:
            predictive_modeling.secondlevel_summary(
                results_dir=result,
                out_dir=model_dir
                )

if __name__ == "__main__":
    run()
