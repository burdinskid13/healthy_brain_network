import warnings
warnings.filterwarnings("ignore")
import os
from hbn.scripts import secondlevel_model, make_firstlevel_model, make_specs
from hbn.scripts import make_model_summary
from hbn.models import train_model as model
from hbn import io
import glob
import shutil
from hbn.constants import Defaults

def firstlevel(feature_spec, target_spec, participant_spec, pydraml_spec, out_dir):
    # print(f'running model with all features', flush=True)

    # make model
    make_firstlevel_model.run(    
                        feature_spec=os.path.join(Defaults.MODEL_SPECS_DIR, feature_spec),
                        target_spec=os.path.join(Defaults.MODEL_SPECS_DIR, target_spec),
                        participant_spec=os.path.join(Defaults.MODEL_SPECS_DIR, participant_spec),
                        pydraml_spec=os.path.join(Defaults.MODEL_SPECS_DIR, pydraml_spec),
                        data_dir=Defaults.INTERIM_FEATURES_DIR,
                        out_dir=out_dir
                        )

    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # train firstlevel model
    features = os.path.join(out_dir, f'features-train.csv')
    model_spec = os.path.join(out_dir, f'model_spec-train.json')
    model.train(
                model_spec=model_spec,
                features=features,
                out_dir=out_dir,
                cache_dir=cache_dir
                )

    # get results file
    results = glob.glob(f'{out_dir}/*out*/*results*.pkl')[0] # should just be one file

    # second level - make summary
    make_model_summary.run(
                    results, # fullpath to results (.pkl)
                    model_spec,
                    out_dir=out_dir,
                    methods=['feature'] # feature interpretability based on feature or permuation importances
                    )

    shutil.rmtree(cache_dir)

def run():
    # define directories
    feature_spec='features-Child-CBCL-spec.json'
    participant_spec='participants-Reading-all-spec.json'
    pydraml_spec='pydraml3-spec.json'
    target_spec='target-Diagnosis-spec.json'
    # out_dir = '/om2/user/maedbh/hbn_data/interim/models/reading_july/test'
    out_dir = '/om2/scratch/tmp/maedbh/HBN_Models/2024-07-30_10-48-23-15'
    cache_dir = None

    # make specs
    # make_specs.run()

    # firstlevel(feature_spec, target_spec, participant_spec, pydraml_spec, out_dir=out_dir)

    for feat in [5,10]:
        for feat_type in ['all-minus-top']:
            
            print(f'running model again with {feat_type} {feat} features', flush=True)

            # define directories
            secondlevel_model_dir = os.path.join(out_dir, f'{feat_type}_{feat}_selected_features')
            secondlevel_model.run(
                firstlevel=os.path.join(out_dir, 'selected_features'),
                secondlevel=secondlevel_model_dir,
                cache_dir=cache_dir,
                feat=feat,
                which_features=feat_type
            )

if __name__ == '__main__':
    run()

