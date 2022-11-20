import click
import warnings
warnings.filterwarnings("ignore")


def preprocess_data():
    import os
    from hbn.constants import Defaults
    from hbn.data import make_dataset

    assessments = ['Child Measures', 'Parent Measures', 'Clinical Measures', 'Teacher Measures']
    parent_file = os.path.join(Defaults, 'data-2022-08-24T16_37_18.263Z.csv')

    for assessment in assessments:
        fdir = os.path.join(Defaults.PHENO_DIR, '_'.join(assessment.split()))
        if not os.path.isdir(fdir):
            make_dataset.parse_phenotypic_data(
                parent_file=parent_file,
                assessment=assessment, 
                out_dir=Defaults.PHENO_DIR
                )

    # creates new clinical diagnosis file
    make_dataset.make_summary()


def make_specs():
    import os
    import glob
    from hbn import io
    from hbn.data import make_dataset
    from hbn.features import build_features
    from hbn.models import first_level_modeling as first_level
    from hbn.constants import Defaults

    # makes test/train splits
    make_dataset.make_train_test_splits(out_dir=Defaults.MODEL_SPEC_DIR)

    # make parent spec file for features
    parent_spec = build_features.make_parent_spec(out_dir=Defaults.FEATURE_DIR)
    
    # make feature and target spec files
    build_features.make_feature_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)
    build_features.make_target_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)

    # makes model specs using feature, target, and participant specs
    # some model functionality (e.g., classifiers) is hardcoded in `hbn.first_level_modeling.make_model_spec`
    features = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*features*.json'))
    targets = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*target*.json'))
    
    # which combination of participants do we want to include in model? all options are saved in Defaults.MODEL_SPEC_DIR
    participants = ["train_participants-ADHD.csv", "train_participants-No_Diagnosis_Given.csv"]
    participants = [os.path.join(Defaults.MODEL_SPEC_DIR, 'train', p) for p in participants]

    # make multiple model specs using features, target, and participant specs 
    for (feature, target) in zip(features, targets):

        # make features dataframe
        feature_path = first_level.make_model_features(feature_spec=feature, 
                                    target_spec=target,
                                    participant_files=participants)

        # make model spec file
        first_level.make_model_spec(features=feature_path,
                                    out_dir=Defaults.MODEL_SPEC_DIR
                                    )


def run_models_first_level(cachedir):
    import glob
    import os
    from hbn import io
    from hbn.constants import Defaults
    from hbn.models import first_level_modeling as first_level

    # grab model specs and run modeling routine
    specs = glob.glob(os.path.join(Defaults.MODEL_SPEC_DIR, '*classifier*DX_01_Cat_binarize*json'))
    for model_spec in specs:

        # load model info
        model_info = io.read_json(model_spec)
        
        # get features & participants
        features = os.path.join(Defaults.FEATURE_DIR, model_info['features_filename'])
        target = os.path.join(Defaults.FEATURE_DIR, model_info['target_filename'])
        participants = os.path.join(Defaults.MODEL_SPEC_DIR, model_info['participants'])

        first_level.run_pipeline(
            model_spec=model_spec, 
            features=features,
            participants=participants,
            cachedir=cachedir, 
            out_dir=Defaults.MODEL_DIR)


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
@click.option("--parse/--no-parse", default=True)
@click.option("--specs/--no-specs", default=True)
@click.option("--run-models-first/--no-run-models-first", default=True)
@click.option("--run-models-second/--no-run-models-second", default=False)

def run(
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/',
    preprocess=True,
    specs=True,
    run_models_first=True,
    run_models_second=True,
    ):
    """ Entire processing workflow for processing phenotypic data from parsing data to running predictive models

    Args: 
        cachedir (str): full path to model cache directory.
        preprocess (bool): parse data from `/nese/mit/group/sig/projects/hbn/phenotype/data-2022-08-24T16_37_18.263Z.csv`. default is True. If data have already been parsed and saved on OpenMind, then parsing isn't re-done. However, nw summary files are created each time this function is run. 
        specs (bool): default is True. Saves feature and model specs (json files) to `/om2/user/maedbh/healthy_brain_network/features` and `/om2/user/maedbh/healthy_brain_network/model_specs`
        run_models_first (bool): default is True. Runs main predictive modeling routine: uses `https://github.com/nipype/pydra-ml` 
        run_models_second (bool): default is True. Wrapper function applied to output from `pydra-ml` to create model summaries, which are saved in `/nese/mit/group/sig/projects/hbn/phenotype/interim/models` 
            on openmind I use: '/home/maedbh/.cache/pydra-ml/cache-wf/'
            on savio I use '/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/'
            on local I use '/Users/maedbhking/pydra-ml/cache-wf/'
    """

    # FIRST STEP
    if preprocess:
       preprocess_data()

    # THIRD STEP
    if specs:
        make_specs()

    # RUNNING MODELS (FIRST LEVEL)
    if run_models_first:
        run_models_first_level(cachedir=cachedir)
    
    # RUNNING MODELS (SECOND LEVEL)
    if run_models_second:
        run_models_second_level()

if __name__ == "__main__":
    run()


    