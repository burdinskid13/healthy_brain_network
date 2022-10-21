import os
import click

@click.command()
@click.option("--spec_file")
@click.option("--cachedir")

def run(
    spec_file='classifier-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize.json',
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/'
    ):
    """
    First Level:
        Runs main predictive modeling routine: uses `https://github.com/nipype/pydra-ml` 
    
    Second Level:
        Wrapper function applied to output from `pydra-ml` to create model summaries, 
        which are saved in `/nese/mit/group/sig/projects/hbn/phenotype/interim/models` 

    Args: 
        spec_file (str): model spec file, contains all configurations that are needed to run modeling routine.
        cachedir (str): full path to model cache directory.
            on openmind I use: '/home/maedbh/.cache/pydra-ml/cache-wf/
            on local I use '/Users/maedbhking/pydra-ml/cache-wf/'
            on savio I use: '/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/'
    """
    from hbn.models import first_level_modeling as first_level
    from hbn.models import second_level_modeling as second_level
    from hbn.constants import Defaults
    from hbn import io

    # load model spec file and get features
    model_spec_dir = os.path.join(Defaults.TEST_DIR, spec_file)
    features = io.read_json(model_spec_dir)['filename']

    # Runs main predictive modeling routine: calls `https://github.com/nipype/pydra-ml` 
    first_level.run_pipeline(spec_file, features, cachedir=cachedir, model_dir=Defaults.MODEL_DIR)

    # # run model pipeline (second level)
    # second_level.get_features()
    # second_level.get_model_summary()



