import warnings
warnings.filterwarnings("ignore")
import os
import glob
import click

from hbn.models import predict_model
from hbn import io
from pathlib import Path

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

    # load models
    fitted_model, feature_names = predict_model.load_model(results)

    # get test data
    model_info = io.load_json(fpath=model_spec)
    X_test, y_test = predict_model.get_test_data(model_dir=model_dir, model_spec=model_info)

    # index test data using model spec
    try:
        X_test = X_test[model_info['x_indices']]
    except:
        X_test = X_test[X_test.columns[model_info['x_indices']]]

    # make predictions
    df_eval, df_pred = predict_model.evaluation(fitted_model, X_test, y_test, feature_names)
    df_eval['model_name'] = Path(model_spec).stem; df_pred['model_name'] = Path(model_spec).stem

    # save model evaluation and predictions to disk
    df_eval.to_csv(os.path.join(model_dir, 'model_evaluation.csv'), index=False)
    df_pred.to_csv(os.path.join(model_dir, 'model_predictions.csv'), index=False)

    print(f'model evaluation and predictions saved to {model_dir}', flush=True)


if __name__ == '__main__':
    run()
