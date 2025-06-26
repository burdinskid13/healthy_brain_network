import glob
import os
from pathlib import Path
from hbn import io
from hbn.models import train_model


def add_columns(df, info):
    """add informative columns to dataframe

    Args:
        df (pd.DataFrame):
        info (dict): model spec info
    """

    def _add_data(df, vars_to_include, info):
        for var in vars_to_include:
            if var in info:
                data =info[var]
            else:
                data = 'all' # use 'all' if not in info
            if not isinstance(data, list):
                data = [data]
            data = [str(d) for d in data]
            df[var] = '_'.join(data)
        return df

    # add feature info
    df['features'] = info['feature_info']['filename'].replace('.csv', '')
    df['feat_spec_name'] = info['feature_info']['spec_name'].replace('-spec', '')
    df['feature_filename'] = info['feature_info']['filename'].replace('.csv', '')

    # add info model spec for participant
    all_vars = list(info['participant_info'].keys())
    vars_to_exclude = ['filename', 'participant_id', 'split', 'spec_name']
    vars_to_include = set(all_vars).difference(set(vars_to_exclude))
    df = _add_data(df, vars_to_include, info=info['participant_info'])

    # add info 
    if 'feature_threshold' in info:
        df['feature_threshold'] = info['feature_threshold']
    
    return df


def run(
    results,
    model_spec,
    out_dir=None,
    methods=['feature'] 
    ):
    """Makes model and feature summary files from results output from `hbn.scripts.run_model`

    Args:
        results (str): fullpath to results file (.pkl)
        model_spec (str): fullpath to model_spec file (.json)
        out_dir (str): directory where modeling summary will be saved
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

        # get clf etc.
        try:
            _, clf, _ =  res[0]['ml_wf.clf_info']
        except:
            _, clf =  res[0]['ml_wf.clf_info']

        # only if data are not permuted
        if not res[0]['ml_wf.permute']:
            for method in methods:
                df = train_model.feature_interpretability(results=res[1], clf=clf)
                if not df.empty: # only save if dataframe is not empty
                    df['model'] = model_name
                    df = add_columns(df, info=spec_info)
                    train_model.save_to_existing_file(dataframe=df, fpath=os.path.join(out_dir, f'{method}_importance.csv'))
                    print('feature summary saved to disk')

    # get model summary (and save to disk)
    model_dataframe = train_model.get_model_metrics(results=data, spec_info=spec_info, clf=clf)
    
    if not model_dataframe.empty: # only save if dataframe is not empty
        model_dataframe['model'] = model_name
        model_dataframe = add_columns(df=model_dataframe, info=spec_info)
        train_model.save_to_existing_file(dataframe=model_dataframe, fpath=os.path.join(out_dir, 'model-summary.csv'))
        print('model summary saved to disk')


if __name__ == "__main__":
    run()