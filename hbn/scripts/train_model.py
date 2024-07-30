import warnings
warnings.filterwarnings("ignore")
import os
from hbn.scripts import secondlevel_model, make_firstlevel_model
from hbn.models import train_model as model
from hbn import io
import click
import glob

def run_model(features, model_spec, model_dir, cache_dir=None):
    """ run model train and model summary

    Args:
        features (str): full path to features file
        model_spec (str): full path to model spec
        model_dir (str): full path to model directory (where `model_spec` and `features` are).
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    from hbn.scripts import train_model, make_model_summary

    print(f'model dir is :{model_dir}', flush=True)

    # define directory where model results will be saved
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # first level - run model
    model.train(
                model_spec=model_spec,
                features=features,
                out_dir=model_dir,
                cache_dir=cache_dir
                )

    # get results file
    results = glob.glob(f'{model_dir}/*out*/*results*.pkl')[0] # should just be one file

    # second level - make summary
    make_model_summary.run(
                    results, # fullpath to results (.pkl)
                    model_spec,
                    out_dir=model_dir,
                    methods=['feature'] # feature interpretability based on feature or permuation importances
                    )

@click.command()
@click.option("--participant_spec", required=True)
@click.option("--target_spec", required=True)
@click.option("--feature_spec", required=True)
@click.option("--pydraml_spec", required=True)
@click.option("--data_dir", required=True)
@click.option("--model_dir", required=True)
@click.option("--cache_dir", required=False)

def run(
    participant_spec,
    target_spec,
    feature_spec,
    pydraml_spec,
    data_dir,
    model_dir,
    cache_dir=None
    ):
    """model routines are created in `hbn.specs.model_specs`


    Args:
        participant_spec (str): filename of participant spec
        target_spec (str): filename of target spec
        feature_spec (str): filename of feature spec
        pydraml_spec (str): filename of pydra-ml spec
        spec_dir (str): directory where `participant_spec`, `target_spec`, `feature_spec`, and `pydraml_spec` are saved
        data_dir (str): directory where `filename` in `feature_spec`, `target_spec`, and `participant_spec` are saved
        model_dir (str): directory where model results will be saved
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """

    print(f'running model with all features', flush=True)

    # train firstlevel model
    # define directories
    selected_features = os.path.join(model_dir, 'selected_features')

    make_firstlevel_model.run(
        feature_spec=feature_spec,
        target_spec=target_spec,
        participant_spec=participant_spec,
        pydraml_spec=pydraml_spec,
        data_dir=data_dir, 
        out_dir=selected_features
    )

    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # train firstlevel model
    features = os.path.join(selected_features, f'features-train.csv')
    model_spec = os.path.join(selected_features, f'model_spec-train.json')
    try:
        run_model(
            features,
            model_spec,
            model_dir=selected_features,
            cache_dir=cache_dir
            )
    except:
        pass

    # train secondlevel model (with selected features from firstlevel model - within CV feature selection is NOT done, model spec is updated to reflect this)
    for feat in [5,10]:
        for feat_type in ['top', 'bottom', 'all-minus-top']:
            
            print(f'running model again with {feat_type} {feat} features', flush=True)

            # define directories
            secondlevel_model_dir = os.path.join(model_dir, f'{feat_type}_{feat}_selected_features')
            try:
                secondlevel_model.run(
                    firstlevel=selected_features,
                    secondlevel=secondlevel_model_dir,
                    cache_dir=cache_dir,
                    feat=feat,
                    which_features=feat_type
                    )   
            except:
                pass

if __name__ == '__main__':
    run()
