import glob
import os
from pathlib import Path
from hbn import io
import pandas as pd
import click

def model_summary(model_path):
    # load data
    df = pd.read_csv(os.path.join(model_path, 'model-summary.csv'), engine='python')

    # model spec
    model_spec = glob.glob(f'{model_path}/*model_spec*')[0] # should just be one spec

    # load json spec file
    info = io.load_json(model_spec)

    # add model name
    df['model_name'] = Path(model_path).name

    # add feature info
    df['features'] = info['feature_info']['filename'].replace('.csv', '')
    df['spec_name'] = info['feature_info']['spec_name'].replace('-spec', '')

    # add info from model spec (VARIABLES ARE SUBJECT TO CHANGE)
    vars_to_add = ['Sex', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round']
    for var in vars_to_add:
        if var in info['participant_info']:
            data = info['participant_info'][var]
        else:
            data = 'all' # use 'all' if not in info
        if not isinstance(data, list):
            data = [data]
        data = [str(d) for d in data]
        df[var] = '_'.join(data)

    return df


def feature_importance(model_path):
    # load data
    df = pd.read_csv(os.path.join(model_path, 'feature_importance.csv'), engine='python')

    # model spec
    model_spec = glob.glob(f'{model_path}/*model_spec*')[0] # should just be one spec

    # load json spec file
    info = io.load_json(model_spec)

    # add model name
    df['model_name'] = Path(model_path).name

    # add feature info
    df['features'] = info['feature_info']['filename'].replace('.csv', '')
    df['spec_name'] = info['feature_info']['spec_name'].replace('-spec', '')

    return df

@click.command()
@click.option("--model_dir", required=True)
def run(
    model_dir
    ):
    """summarize models into one csv file that will be saved to `data/processed/model_summary.csv`. This file can then be visualized

    Args:
        model_dir (str): full path to model directory
    """

    # load all models
    models = glob.glob(os.path.join(f'{model_dir}/*/'))

    # load model summary from all directories in `model_dir`
    df_summary_all = pd.DataFrame(); df_feature_all = pd.DataFrame()
    df_pred_all = pd.DataFrame(); df_eval_all = pd.DataFrame()
    for model_path in models:

        # get model summary and feature importance
        df_summary = model_summary(model_path)
        df_feature = feature_importance(model_path)
        df_pred = model_predictions(model_path)
        df_eval = model_evaluation(model_path)

        df_summary_all = pd.concat([df_summary_all, df_summary])
        df_feature_all = pd.concat([df_feature_all, df_feature])
        df_pred_all = pd.concat([df_pred_all, df_pred])
        df_eval_all = pd.concat([df_eval_all, df_eval])

    # save out dataframes
    df_summary_all.to_csv(os.path.join(model_dir, 'overall_model_summary.csv'), index=False)
    df_feature_all.to_csv(os.path.join(model_dir, 'overall_feature_importance.csv'), index=False)
    df_pred_all.to_csv(os.path.join(model_dir, 'overall_predictions.csv'), index=False)
    df_eval_all.to_csv(os.path.join(model_dir, 'overall_evaluation.csv'), index=False)

if __name__ == "__main__":
    run()
