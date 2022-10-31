import os
import click
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


def run_model_pipeline_secondlevel():
    """Makes model and feature summary files from results output from `run_model_pipeline_firstlevel`

    Saves output in `../interim/models/`
    """
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
@click.option("--parse-data/--no-parse-data", default=False)
@click.option("--feature-specs/--no-feature-specs", default=False)
@click.option("--model-specs/--no-model-specs", default=False)
@click.option("--run-models-first/--no-run-models-first", default=True)
@click.option("--run-models-second/--no-run-models-second", default=False)

def run(
    cachedir='/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/',
    parse_data=False,
    target_specs=False,
    feature_specs=False,
    model_specs=False,
    run_models_first=True,
    run_models_second=True,
    ):
    """ Entire processing workflow for processing phenotypic data from parsing data to running predictive models

    Args: 
        cachedir (str): full path to model cache directory.
        parse_data (bool): parse data from `/nese/mit/group/sig/projects/hbn/phenotype/data-2022-08-24T16_37_18.263Z.csv`. default is False because data have already been parsed and saved on OpenMind.
        feature_specs (bool): default is True. Saves feature specs (json and csv files) to `/om2/user/maedbh/healthy_brain_network/features`
        model_specs (bool): default is True. Make model specs (json files) and save to `/om2/user/maedbh/healthy_brain_network/models`
        run_models_first (bool): default is True. Runs main predictive modeling routine: uses `https://github.com/nipype/pydra-ml` 
        run_models_second (bool): default is True. Wrapper function applied to output from `pydra-ml` to create model summaries, which are saved in `/nese/mit/group/sig/projects/hbn/phenotype/interim/models` 
            on openmind I use: '/home/maedbh/.cache/pydra-ml/cache-wf/
            on savio I use '/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/'
            on local I use '/Users/maedbhking/pydra-ml/cache-wf/'
    """
    from hbn import io
    import os
    import glob
    from hbn.constants import Defaults
    from hbn.data import make_dataset
    from hbn.features import build_features
    from hbn.models import first_level_modeling as first_level
    from hbn.models import second_level_modeling as second_level

    # make cachedir if it doesn't exist
    io.make_dirs(cachedir)

    # FIRST STEP
    if parse_data:
        make_dataset.parse_phenotypic_data(assessment=['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures'])
        make_dataset.make_summary()

    # SECOND STEP
    if feature_specs:
        parent_spec = os.path.join(Defaults.FEATURE_DIR, 'features-parent_spec.json')
        feature_fpaths = build_features.make_spec_files(parent_spec, out_dir=Defaults.FEATURE_DIR)
        for feature_spec in feature_fpaths:
            build_features.make_feature_files(feature_spec, out_dir=Defaults.FEATURE_DIR)

    # FOURTH STEP
    if model_specs:
        # grab feature specs and make model specs
        fpaths = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*.json'))
        for fpath in fpaths:
            try:
                first_level.make_specs(feature_spec=fpath, out_dir=Defaults.MODEL_SPEC_DIR)
            except:
                pass

    # RUNNING MODELS (FIRST LEVEL)
    if run_models_first:
        # grab model specs and run modeling routine
        specs = glob.glob(os.path.join(Defaults.MODEL_SPEC_DIR, '*classifier*DX_01_Cat_binarize*json'))
        for model_spec in specs:
            first_level.run_pipeline(model_spec, cachedir=cachedir, out_dir=Defaults.MODEL_DIR)
    
    # RUNNING MODELS (SECOND LEVEL)
    if run_models_second:
        second_level.get_features()
        second_level.get_model_summary()

if __name__ == "__main__":
    run()


    