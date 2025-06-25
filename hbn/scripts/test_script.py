import warnings
warnings.filterwarnings("ignore")
import os
import glob
import pandas as pd
from hbn.scripts import secondlevel_model, make_firstlevel_model, make_specs
from hbn.scripts import make_model_summary
from hbn.models import train_model as model
from hbn.constants import Defaults
import shutil

def run():
    # define directories
    feature_spec='features-cbcl-spec.json' #'features-all-questions-spec.json'
    participant_spec="participants-adhd-all-male-spec.json"
    pydraml_spec='pydraml3-spec.json'
    target_spec='target-Diagnosis-ADHD-spec.json'
    out_dir = '/orcd/data/satra/001/users/maedbh/hbn_data/interim/models/adhd_dx_TEST/'
    release = 'Release11_Apr2024'
    cache_dir = None

    # make specs
    make_specs.run()

    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)


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

    # check summary results
    df = pd.read_csv(os.path.join(out_dir, 'model-summary.csv'))

    # get data and print summary
    df1 = df[df['data']=='model-data']
    print(df1[['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']].mean())


if __name__ == '__main__':
    run()

