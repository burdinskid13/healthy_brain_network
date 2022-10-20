import os
import click

@click.command()
@click.option("--cachedir")

def run(
    cachedir='/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/',
    ):
    """
    First Level:
        Runs main predictive modeling routine: uses `https://github.com/nipype/pydra-ml` 
    
    Second Level:
        Wrapper function applied to output from `pydra-ml` to create model summaries, 
        which are saved in `/nese/mit/group/sig/projects/hbn/phenotype/interim/models` 

    Args: 
        cachedir (str): full path to model cache directory.
            on openmind I use: '/home/maedbh/.cache/pydra-ml/cache-wf/
            on local I use '/Users/maedbhking/pydra-ml/cache-wf/'
    """
    from hbn.models import first_level_modeling as first_level
    from hbn.models import second_level_modeling as second_level

    # Runs main predictive modeling routine: calls `https://github.com/nipype/pydra-ml` 
    first_level.run_pipeline(cachedir=cachedir)

    # run model pipeline (second level)
    second_level.get_features()
    second_level.get_model_summary()



