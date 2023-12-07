
import glob
import os
from pathlib import Path
from src import io
from src.models import train_model

def run(
    results,
    model_spec,
    out_dir=None,
    methods=['feature'] 
    ):
    """Makes model and feature summary files from results output from `src.scripts.run_model`

    Args:
        results (str): fullpath to results file (.pkl)
        model_spec (str): fullpath to model_spec file (.json)
        out_dir (str): directory where second level modeling summary will be saved
        methods (list of str): feature interpretability based on feature or permuation importances. default is ['feature']
    Returns:
        Saves summary in `out_dir`
    """

    # make model out_dir if it doesn't already exist
    io.make_dirs(out_dir)

    # load results
    data, spec_info = train_model.load_results(results=results, spec_file=model_spec)
    model_name = Path(results).stem.split('-')[1]

    # loop over results and get feature and permuation importances
    for res in data:
        # only if data are not permuted
        if not res[0]['ml_wf.permute']:
            for method in methods:
                df = train_model.feature_interpretability(results=res[1], spec_info=spec_info, method=method)
                if not df.empty: # only save if dataframe is not empty
                    df['model'] = model_name
                    train_model.save_to_existing_file(dataframe=df, fpath=os.path.join(out_dir, f'{method}_importance.csv'))
                    print('feature summary saved to disk')

    # get model summary (and save to disk)
    model_dataframe = train_model.get_model_metrics(results=data, spec_info=spec_info)
    
    if not model_dataframe.empty: # only save if dataframe is not empty
        model_dataframe['model'] = model_name
        train_model.save_to_existing_file(dataframe=model_dataframe, fpath=os.path.join(out_dir, 'model-summary.csv'))
        print('model summary saved to disk')


if __name__ == "__main__":
    run()