import os


def load_results(fpath):
    """load results from `results-<modelname>.pkl` file

    Args: 
        fpath (str): full path to results file
    Returns:
        results (list of dict)
    """
    import pickle as pk

    with open(fpath, "rb") as fp:
        results = pk.load(fp)
    
    return results


def get_features(
    results, 
    model,
    feature_importance=True, 
    permutation_importance=False
    ):
    """get feature and permutation importances if they exist in `results`

    Args: 
        results (list of dict): data loaded from `out-localspec-<modelname>/results-<modelname>.pkl` (output from pydra-ml modeling routine)
    Returns: 
        df_feature_importance (pd dataframe), df_permutation_importance (pd dataframe)
    """
    import pandas as pd

    def _add_modelname(df, model, results):
        df['model'] = model
        df['clf'] = results['ml_wf.clf_info'][1]
        return df

    df_feature_importance = pd.DataFrame(); df_permutation_importance = pd.DataFrame()
    for res in results:
        
        wf_permute = res[0]['ml_wf.permute']
        if not wf_permute:

            # get feature importance
            if feature_importance:
                df_feature_importance = _get_feature_importance(model_output=res[1].output)
                df_feature_importance = _add_modelname(df_feature_importance, model, results=res[0])

            # get permutation importance
            if permutation_importance:
                df_permutation_importance = _get_permutation_importance(model_output=res[1].output)
                df_permutation_importance = _add_modelname(df_permutation_importance, model, results=res[0])

    return df_feature_importance, df_permutation_importance


def get_model_summary(results, spec_file):
    """get model summary for `results`. code has only been tested on results which have one classifier.

    Args: 
        results (list of dict): results output from `load_results`
        spec_file (str): full path to *.json spec file for each model. stored in `out-localspec-<modelname>`
    Returns:
        df_all (pd dataframe)
    """
    from hbn import io
    from pathlib import Path
    import pandas as pd
    import numpy as np

    # load spec info from file
    spec_info = io.read_json(spec_file)

    def _unpack_features(dataframe):
        for idx,col in enumerate(['Assessment', 'Domain', 'Measure']):
            dataframe[col] = dataframe['features'].str.split('-').str.get(idx)
        return dataframe

    df_all = pd.DataFrame()
    for res in results:

        # scores
        permute = res[0]['ml_wf.permute']
        data = 'model-data'
        if permute:
            data = 'model-null'

        # make dataframe
        df = pd.DataFrame(np.array(res[1].output.score), columns=spec_info['metrics'])
        df['data'] = data
        df['clf'] = res[0]['ml_wf.clf_info'][1]
        df['target'] = spec_info['target_vars'][0]
        df['features'] = '-'.join(spec_info['filename'].split('-')[1:-1]) 
        df['splits'] = df.index
        df = _unpack_features(df)

        df_all = pd.concat([df_all, df])
    
    return df_all


def _get_feature_importance(model_output):
    """get feature importances across splits

    Args:
        model_output (dict): `results.output` len(list) = # of splits
    """
    import numpy as np  
    import pandas as pd

    df_features = pd.DataFrame()

    # extract feature importance
    feature_splits = np.array(model_output.feature_importance)
    feature_names = np.array(model_output.feature_names)

    n_splits, n_feats = feature_splits.shape

    if n_feats==len(feature_names):

        feature_names_mat = np.tile(np.reshape(feature_names, (n_feats,1)), n_splits).T
        feature_splits_sort_idx = np.argsort(feature_splits)

        features_sorted = np.take_along_axis(feature_names_mat, feature_splits_sort_idx, axis=1)
        features_sorted = features_sorted[:,::-1] # reverse order

        # get features across splits
        df_rank = _rank_order_features_across_splits(dataframe=pd.DataFrame(features_sorted))
        df_common = _most_commonly_occuring_features(dataframe=pd.DataFrame(features_sorted))

        df_features = pd.concat([df_rank, df_common], axis=1)

    return df_features


def _get_permutation_importance(model_output):
    """get feature permuation across splits

    Args:
        model_output (dict): `results.output` len(list) = # of splits
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