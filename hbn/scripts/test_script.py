import warnings
warnings.filterwarnings("ignore")
import os
from hbn.scripts import secondlevel_model, make_firstlevel_model, make_specs
from hbn.scripts import make_model_summary
from hbn.models import train_model as model
from hbn.constants import Defaults

def run():
    # define directories
    feature_spec='features-all-questions-spec.json'
    participant_spec="participants-adhd-all-male-spec.json"
    pydraml_spec='pydraml3-spec.json'
    target_spec='target-Diagnosis-ADHD-spec.json'
    out_dir = '/orcd/data/satra/001/users/maedbh/hbn_data/interim/models/adhd_dx_TEST/'
    release = 'Release11_Apr2024'
    cache_dir = None

    # make specs
    # make_specs.run()

    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # make model
    make_firstlevel_model.run(    
                        feature_spec=os.path.join(Defaults.MODEL_SPECS_DIR, feature_spec),
                        target_spec=os.path.join(Defaults.MODEL_SPECS_DIR, target_spec),
                        participant_spec=os.path.join(Defaults.MODEL_SPECS_DIR, participant_spec),
                        pydraml_spec=os.path.join(Defaults.MODEL_SPECS_DIR, pydraml_spec),
                        data_dir=os.path.join(Defaults.INTERIM_FEATURES_DIR, release),
                        out_dir=out_dir
                        )

    # train firstlevel model
    features = os.path.join(out_dir, f'features-train.csv')
    model_spec = os.path.join(out_dir, f'model_spec-train.json')
    model.train(
                model_spec=model_spec,
                features=features,
                out_dir=out_dir,
                cache_dir=cache_dir
                )

    # # get results file
    # results = glob.glob(f'{out_dir}/*out*/*results*.pkl')[0] # should just be one file

    # # second level - make summary
    # make_model_summary.run(
    #                 results, # fullpath to results (.pkl)
    #                 model_spec,
    #                 out_dir=out_dir,
    #                 methods=['feature'] # feature interpretability based on feature or permuation importances
    #                 )

    # shutil.rmtree(cache_dir)


if __name__ == '__main__':
    run()

