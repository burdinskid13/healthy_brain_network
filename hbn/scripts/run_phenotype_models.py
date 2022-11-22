import click
import warnings
warnings.filterwarnings("ignore")


def run_models_second_level():
    """Makes model and feature summary files from results output from `run_model_pipeline_firstlevel`

    Saves output in `../interim/models/`
    """
    import os
    import glob
    from pathlib import Path
    from hbn.constants import Defaults
    from hbn.models import second_level_modeling as second_level

    # grab list of models
    model_dirs = glob.glob(os.path.join(Defaults.MODEL_DIR, '*out-localspec*/*results*.pkl*'))
    spec_dirs = glob.glob(os.path.join(Defaults.MODEL_DIR, '*out-localspec*/*.json*'))
    
    for (model_fpath, spec_fpath) in zip(model_dirs, spec_dirs):

        # get model name
        modelname = Path(model_fpath).stem.split('-')[1] # `results-<modelname>`
        clf = Path(spec_fpath).name.split('-')[0]

        # get outpaths
        model_outpath = os.path.join(Defaults.MODEL_DIR, f'{clf}-all-phenotypic-models-performance.csv')
        feat_outpath = os.path.join(Defaults.MODEL_DIR, f'{clf}-feature_importance.csv')
        perm_outpath = os.path.join(Defaults.MODEL_DIR, f'{clf}-permutation_importance.csv')

        # load results
        results, spec_info = second_level.load_results(model_file=model_fpath, spec_file=spec_fpath)

        # make feature summary (and save to disk)
        feature_df, permutation_df = second_level.get_features(
            results=results, 
            spec_info=spec_info,
            feature_importance=True, 
            permutation_importance=False
            )
        feature_df['model'] = modelname; permutation_df['model'] = modelname

        # get model summary (and save to disk)
        model_dataframe = second_level.get_model_summary(
                        results=results, 
                        spec_info=spec_info, 
                        )
        model_dataframe['model'] =  modelname

        # save to disk
        _save_to_existing_file(dataframe=feature_df, fpath=feat_outpath)
        _save_to_existing_file(dataframe=permutation_df, fpath=perm_outpath)
        _save_to_existing_file(dataframe=model_dataframe, fpath=model_outpath)


@click.command()
@click.option("--cachedir")

def run(
    cachedir='/om2/user/maedbh/.cache/pydra-ml/cache-wf/',
    ):
    """run first level modeling pipeline

    Args: 
        model_spec (str): model spec filename (not full path)
        cachedir (str): full path to  cache directory for pydra-ml intermediary outputs
    """
    import glob
    import os
    from hbn.constants import Defaults
    from hbn.models import first_level_modeling as first_level

    specs = glob.glob(os.path.join(Defaults.MODEL_SPEC_DIR, '*classifier*'))

    # loop over model specs
    for model_spec in specs:
        first_level.run_pipeline(
            model_spec=model_spec, 
            spec_dir=Defaults.MODEL_SPEC_DIR, 
            out_dir=Defaults.MODEL_DIR,
            cachedir=cachedir
            )

if __name__ == "__main__":
    run()


    