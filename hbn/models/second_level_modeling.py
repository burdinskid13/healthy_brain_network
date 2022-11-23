import os
from hbn.constants import Defaults

def run_pipeline(
    results_dir,
    out_dir=Defaults.MODEL_DIR
    ):
    """Makes model and feature summary files from results output from `run_model_pipeline_firstlevel`

    Saves output in `../interim/models/`

    Args:
        results_dir (str): fullpath to top-level results dir. for example '../interim/models/out-localspec-<>'
        out_dir (str): directory where second level modeling summary will be saved
    """
    import glob
    import os
    from pathlib import Path

    # get results file
    model_name = Path(results_dir).stem
    results_file = os.path.join(results_dir, f'results-{model_name}.pkl')

    # get model spec file
    spec_file = glob.glob(os.path.join(results_dir, '*.json'))[0]

    # load results
    results, spec_info = load_results(results=results_file, spec_file=spec_file)
    clf = Path(spec_file).name.split('-')[0] # 'classifier' or 'regression'

    # loop over results and get feature and permuation importances
    for res in results:
        # get feature importances
        df_features = get_feature_importance(results=res, spec_info=spec_info)
        feature_fname = f'{clf}-feature_importance.csv'
        _save_to_existing_file(model_name, dataframe=df_features, fpath=os.path.join(out_dir, feature_fname))

        # get permuation importances
        df_permutation = get_permutation_importance(results=res, spec_info=spec_info)
        permutation_fname = f'{clf}-permutation_importance.csv'
        _save_to_existing_file(model_name, dataframe=df_permutation, fpath=os.path.join(out_dir, permutation_fname))

    # get model summary (and save to disk)
    model_dataframe = make_model_summary(
                    results=results, 
                    spec_info=spec_info, 
                    )
    model_fname = f'{clf}-all-phenotypic-models-performance.csv'
    _save_to_existing_file(model_name, dataframe=model_dataframe, fpath=os.path.join(out_dir, model_fname))


def load_results(results_file, spec_file):
    """load results from `results-<modelname>.pkl` file

    Args: 
        results_file (str): full path to results file
        spec_file (str): full path to model spec file
    Returns:
        results (list of dict)
    """
    import pickle as pk
    from hbn import io

    with open(results_file, "rb") as fp:
        results = pk.load(fp)

    # load spec info from file
    spec_info = io.read_json(spec_file)
    
    return results, spec_info


def _add_model_parameters(dataframe, model_name, spec_info, results):
    """add model parameters to dataframe
    """
    from pathlib import Path

    # get modelname
    dataframe['model'] = model_name
    dataframe['clf'] = results['ml_wf.clf_info'][1]
    dataframe['target'] = spec_info['target_vars'][0]
    dataframe['features'] = '-'.join(spec_info['filename'].split('-')[1:-1]) 
    for idx,col in enumerate(['Assessment', 'Domain', 'Measure']):
        dataframe[col] = dataframe['features'].str.split('-').str.get(idx)

    return dataframe


def make_model_summary(model_name, results, spec_info):
    """get model summary for `results`. code has only been tested on results which have one classifier.

    Args: 
        model_name (str): model name
        results (list of dict): results output from `load_results`
        spec_file (str): full path to *.json spec file for each model. stored in `out-localspec-<modelname>`
    Returns:
        df_all (pd dataframe)
    """
    from hbn import io
    from pathlib import Path
    import pandas as pd
    import numpy as np

    df_all = pd.DataFrame()
    for res in results:

        # scores
        permute = res[0]['ml_wf.permute']
        data = 'model-null'
        if permute:
            data = 'model-data'

        # make dataframe
        df = pd.DataFrame(np.array(res[1].output.score), columns=spec_info['metrics'])
        df['data'] = data
        df['splits'] = df.index
        df = _add_model_parameters(df, model_name, spec_info=spec_info, results=res[0])

        df_all = pd.concat([df_all, df])
    
    return df_all


def get_feature_importance(model_name, results, spec_info):
    """get feature importances across splits

    Args:
        results (list of dict): 
    """
    import numpy as np  
    import pandas as pd

    df_features = pd.DataFrame()

    # extract feature importance
    feature_splits = np.array(results[1].output.feature_importance)
    feature_names = np.array(results[1].output.feature_names)

    n_splits, n_feats = feature_splits.shape

    if n_feats==len(feature_names):

        feature_names_mat = np.tile(np.reshape(feature_names, (n_feats,1)), n_splits).T
        feature_splits_sort_idx = np.argsort(feature_splits)

        features_sorted = np.take_along_axis(feature_names_mat, feature_splits_sort_idx, axis=1)
        features_sorted = features_sorted[:,::-1] # reverse order

        # get features across splits
        df_rank = _rank_order_features_across_splits(dataframe=pd.DataFrame(features_sorted))
        df_common = _most_commonly_occuring_features(dataframe=pd.DataFrame(features_sorted))
        df_sum = _sum_feature_weights(feature_splits, feature_names)

        df_features = pd.concat([df_rank, df_common, df_sum], axis=1)

    df_features = _add_model_parameters(df_features, model_name, spec_info=spec_info, results=results[0])

    return df_features


def get_permutation_importance(model_name, results, spec_info):
    """get feature permuation across splits

    Args:
        results (list of dict): 
    """
    pass


def _rank_order_features_across_splits(dataframe):
    """ rank orders features by how commonly they occur within a split, keeping each entry unique (as far as possible)

    Args: 
        dataframe (pd dataframe): each column is a feature and rows are n splits
    """
    import pandas as pd
    import numpy as np

    feature_importances = []; feature_probabilities = [];
    for col in dataframe.columns:
        idx = 0
        feat = dataframe[col].value_counts().index[idx]
        val = dataframe[col].value_counts().values[idx] / len(dataframe)
        while feat in feature_importances:
            try:
                feat = dataframe[col].value_counts().index[idx+1]
                val = dataframe[col].value_counts().values[idx+1] / len(dataframe)
            except:
                break
            idx += 1
        feature_importances.append(feat)
        feature_probabilities.append(val)
        print(f'adding {feat} to list')
    df_features = pd.DataFrame(feature_importances, columns=['feature_names_rank_order'])
    df_features['feature_probabilities_rank_order'] = feature_probabilities
    
    return df_features 


def _most_commonly_occuring_features(dataframe):
    """ returns most commonly occuring feature per split, does not enforce unique values through rank ordering

    Args: 
        dataframe (pd dataframe): each column is a feature and rows are n splits
    """
    import pandas as pd

    feature_importances = []; feature_probabilities = [];
    for col in dataframe.columns:
        idx = 0
        feat = dataframe[col].value_counts().index[idx]
        val = dataframe[col].value_counts().values[idx] / len(dataframe)
        feature_importances.append(feat)
        feature_probabilities.append(val)
        print(f'adding {feat} to list')
    df_features = pd.DataFrame(feature_importances, columns=['feature_names_common'])
    df_features['feature_probabilities_common'] = feature_probabilities
    
    return df_features 


def _sum_feature_weights(feature_splits, feature_names):
    import numpy as np
    import pandas as pd

    # sum up weights for each feature (across splits)
    feature_sum = np.sum(feature_splits,0)

    # rank order summed weights
    sort_idx = np.argsort(feature_sum)

    df = pd.DataFrame()
    df['feature_sum'] = feature_sum[sort_idx[::-1]]
    df['feature_names_sum'] = np.array(feature_names)[sort_idx[::-1]]

    return df


def _save_to_existing_file(dataframe, fpath):
    import pandas as pd

    df = pd.DataFrame()
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
        except:
            pass
    df_out = pd.concat([df, dataframe])
    df_out.to_csv(fpath, index=False)

