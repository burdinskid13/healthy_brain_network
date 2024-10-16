import warnings
warnings.filterwarnings("ignore")
import os
from hbn.scripts import secondlevel_model, make_firstlevel_model
from hbn.models import train_model as model
from hbn import io
import click
import glob

def update_spec(spec, threshold='all'):
    """update pydra-ml spec file with new model features
    spec (str): full path to pydra-ml spec
    threshold (str): 'all' or 'top' or 'bottom'
    """
    info = io.load_json(spec)
    info['feature_threshold'] = threshold

    # save out spec
    io.save_json(spec, info)


def firstlevel(model_dir, feature_spec, target_spec, participant_spec, pydraml_spec, data_dir, cache_dir):
    print(f'running model with all features', flush=True)

    make_firstlevel_model.run(
        feature_spec=feature_spec,
        target_spec=target_spec,
        participant_spec=participant_spec,
        pydraml_spec=pydraml_spec,
        data_dir=data_dir, 
        out_dir=model_dir
    )

    # train firstlevel model
    features = os.path.join(model_dir, f'features-train.csv')
    model_spec = os.path.join(model_dir, f'model_spec-train.json')

    # update model spec
    update_spec(spec=model_spec, threshold='selected_all')

    try:
        run_model(
            features,
            model_spec,
            model_dir=model_dir,
            cache_dir=cache_dir
            )
    except:
        pass


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
        participant_spec (str): fullpath to participant spec
        target_spec (str): fullpath to target spec
        feature_spec (str): fullpath to feature spec
        pydraml_spec (str): fullpath to pydra-ml spec
        data_dir (str): directory where `filename` in `feature_spec`, `target_spec`, and `participant_spec` are saved
        model_dir (str): directory where model results will be saved
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    """
    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'
        
    # run first level
    firstlevel_dir = os.path.join(model_dir, 'selected_all')
    firstlevel(firstlevel_dir, 
                feature_spec, 
                target_spec, 
                participant_spec, 
                pydraml_spec, 
                data_dir,
                cache_dir
                )

    # train secondlevel model (with selected features from firstlevel model - within CV feature selection is NOT done, model spec is updated to reflect this)
    for feat in [5,10]:
        for feat_type in ['top', 'bottom']: # 'all-minus-top'
            
            print(f'running model again with {feat_type} {feat} features', flush=True)

            # define directories
            secondlevel_dir = os.path.join(model_dir, f'{feat_type}_{feat}_selected_features')
            try:
                secondlevel_model.run(
                    firstlevel=firstlevel_dir,
                    secondlevel=secondlevel_dir,
                    cache_dir=cache_dir,
                    feat=feat,
                    which_features=feat_type
                    )   
            except:
                pass

if __name__ == '__main__':
    run()
