import click

@click.command()
@click.option("--feature_spec")
@click.option("--cachedir")

def run(
    feature_spec='features-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize.json',
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/'
    ):
    """
    FIRST STEP: 
        makes 'participants.csv' and 'Clinical_Diagnosis_Demographics.csv' that are used in later modeling routine
    SECOND STEP: 
        make feature csv file using `feature_spec`
    THIRD STEP: 
        make participants file using `first_level_modeling.make_train_test_splits`
    FOURTH STEP:
        make model spec file using `feature_spec` and `participants` and other information hardcoded in `first_level_modeling.make_model_spec`
    FIFTH STEP:
        run modeling routine: uses `https://github.com/nipype/pydra-ml` 
    
    Args: 
        feature_spec (str): feature spec file, contains all configurations that are needed to create feature files, model spec file, and run modeling routine.
        cachedir (str): full path to model cache directory.
            on openmind I use: '/home/maedbh/.cache/pydra-ml/cache-wf/
            on local I use '/Users/maedbhking/pydra-ml/cache-wf/'
            on savio I use: '/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/'
    """
    import os
    from hbn import io
    import pandas as pd
    from hbn.constants import Defaults
    from hbn.data import make_dataset
    from hbn.features import build_features
    from hbn.models import first_level_modeling as first_level

    # make cachedir if it doesn't exist
    io.make_dirs(cachedir)

    # make summary files 
    make_dataset.make_summary()
    make_dataset.make_train_test_splits()

    TEST_DATA = os.path.join(Defaults.TEST_DIR, 'test_data')

    # Make feature csv
    features = build_features.make_feature_files(feature_spec, out_dir=TEST_DATA)

    # make model spec file
    model_spec = first_level.make_model_spec(feature_spec, participants='train_participants-ADHD.csv', out_dir=TEST_DATA)
    model_info = io.read_json(model_spec)

    # Run main predictive modeling routine: calls `https://github.com/nipype/pydra-ml` 
    first_level.run_pipeline(
            model_spec=model_spec, 
            features=features,
            participants=os.path.join(TEST_DATA, model_info['participants']),
            cachedir=cachedir, 
            out_dir=Defaults.MODEL_DIR)

if __name__ == "__main__":
    run()



