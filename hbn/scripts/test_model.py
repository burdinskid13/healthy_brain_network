import warnings
warnings.filterwarnings("ignore")
import os
import glob
import click
import pandas as pd

from hbn.models import predict_model
from hbn import io
from pathlib import Path

def add_columns(df, info):
    """add informative columns to dataframe

    Args:
        df (pd.DataFrame):
        info (dict): model spec info
    """

    # add feature info
    df['features'] = info['feature_info']['filename'].replace('.csv', '')
    df['feat_spec_name'] = info['feature_info']['spec_name'].replace('-spec', '')
    df['model_type'] = info['feature_info']['model_type']

    # add info from model spec (VARIABLES ARE SUBJECT TO CHANGE)
    vars_to_include = ['Age_round', 'Sex', 'DX_Cat_Name', 'PreInt_Demos_Fam,Child_Race_cat', 'spec_name']
    for var in vars_to_include:
        if var in info['participant_info']:
            data = info['participant_info'][var]
        else:
            data = 'all' # use 'all' if not in info
        if not isinstance(data, list):
            data = [data]
        data = [str(d) for d in data]
        df[var] = '_'.join(data)
    
    return df

@click.command()
@click.option("--model_dir", required=True)
@click.option("--model_spec", required=True)

def run(model_dir, model_spec):
    """run evaluation on new test data

    Args:
        model_dir (str): fullpath to model directory
        model_spec (str): full path to model spec
    """
    results = glob.glob(f'{model_dir}/*out*/*results*.pkl')[0] # should just one results file
    model_name = Path(results).stem.split('-')[1]

    # load models
    fitted_model, feature_names = predict_model.load_model(results)

    # get test data
    model_info = io.load_json(fpath=model_spec)
    X_test, y_test = predict_model.get_test_data(model_dir=model_dir, model_spec=model_info)

    if X_test.shape[1]!=len(feature_names):
        X_test_df = pd.DataFrame(np.zeros((X_test.shape[0], len(feature_names))), columns=feature_names)
        for col in X_test_df.columns:
            if col in X_test.columns:
                X_test_df.loc[:, col] = X_test[col]
    else:
        X_test_df = X_test

    # how many features?
    num_features = len(model_info['x_indices'])

    # make predictions
    df_eval, df_pred = predict_model.evaluation(fitted_model, X_test, y_test, feature_names)
    df_eval['model'] = model_name; df_pred['model'] = model_name
    df_eval['number_of_features'] = num_features; df_pred['number_of_features'] = num_features
    df_eval = add_columns(df=df_eval, info=model_info); df_pred = add_columns(df=df_pred, info=model_info)
    
    # save model evaluation and predictions to disk
    df_eval.to_csv(os.path.join(model_dir, 'model_evaluation.csv'), index=False)
    df_pred.to_csv(os.path.join(model_dir, 'model_predictions.csv'), index=False)

    print(f'model evaluation and predictions saved to {model_dir}', flush=True)


if __name__ == '__main__':
    run()
