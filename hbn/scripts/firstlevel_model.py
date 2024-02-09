import warnings
warnings.filterwarnings("ignore")
import click
from hbn.scripts import make_firstlevel_model, run_model
import os

@click.command()
@click.option("--participant_spec", required=True)
@click.option("--target_spec", required=True)
@click.option("--feature_spec", required=True)
@click.option("--pydraml_spec", required=True)
@click.option("--data_dir", required=True)
@click.option("--spec_dir", required=True)
@click.option("--model_dir", required=True)
@click.option("--cache_dir", required=False)
def run(
    participant_spec,
    target_spec,
    feature_spec,
    pydraml_spec,
    data_dir,
    spec_dir, 
    model_dir,
    cache_dir=None
    ):
    """train firstlevel model and get summary of model results

    Args:
        participant_spec (str): filename of participant spec
        target_spec (str): filename of target spec
        feature_spec (str): filename of feature spec
        pydraml_spec (str): filename of pydra-ml spec
        spec_dir (str): directory where `participant_spec`, `target_spec`, `feature_spec`, and `pydraml_spec` are saved
        data_dir (str): directory where `filename` in `feature_spec`, `target_spec`, and `participant_spec` are saved
        model_dir (str): directory where model results will be saved
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    Returns:
        saves pickled model to `model_dir`
    """

    # make model spec and features
    features, model_spec = make_firstlevel_model.run(
                            feature_spec=os.path.join(spec_dir, feature_spec),  
                            target_spec=os.path.join(spec_dir, target_spec),
                            participant_spec=os.path.join(spec_dir, participant_spec),
                            pydraml_spec=os.path.join(spec_dir, pydraml_spec),
                            data_dir=data_dir,
                            out_dir=model_dir
                            )
                        
    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'


    # train firstlevel model
    run_model.run(features,
                  model_spec,
                  model_dir=model_dir,
                  cache_dir=cache_dir
                  )

if __name__ == '__main__':
    run()