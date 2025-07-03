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
import click

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

    # make specs
    make_specs.run()

    # define cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # if os.path.isdir(cache_dir):
    #     shutil.rmtree(cache_dir)
    # if os.path.isdir(model_dir):
    #     shutil.rmtree(model_dir)

    # make model
    make_firstlevel_model.run(    
                        feature_spec=feature_spec,
                        target_spec=target_spec,
                        participant_spec=participant_spec,
                        pydraml_spec=pydraml_spec,
                        data_dir=data_dir,
                        out_dir=model_dir,
                        )

    # # train firstlevel model
    features = os.path.join(model_dir, f'features-train.csv')
    model_spec = os.path.join(model_dir, f'model_spec-train.json')
    model.train(
                model_spec=model_spec,
                features=features,
                out_dir=model_dir,
                cache_dir=cache_dir
                )

    # # get results file
    results = glob.glob(f'{model_dir}/*out*/*results*.pkl')[0] # should just be one file

    # second level - make summary
    make_model_summary.run(
                    results, # fullpath to results (.pkl)
                    model_spec,
                    out_dir=model_dir,
                    methods=['feature'] # feature interpretability based on feature or permuation importances
                    )

    # shutil.rmtree(cache_dir)

    # check summary results
    df = pd.read_csv(os.path.join(model_dir, 'model-summary.csv'))

    # get data and print summary
    df1 = df[df['data']=='model-data']
    print(df1[['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']].mean())


if __name__ == '__main__':
    run()

