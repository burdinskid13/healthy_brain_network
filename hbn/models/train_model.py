

def feature_interpretability(results, spec_info, method='feature'):
    import pandas as pd

    df1 = order_across_splits(results=results, method=method)
    df2 = model_based_importance(results=results)

    # concat into features dataframe
    df_features = pd.concat([df1, df2], axis=1)

    return df_features


def load_results(results, spec_file):
    """load results from `results-<modelname>.pkl` file

    Args: 
        results (str): full path to results file
        spec_file (str): full path to model spec file
    Returns:
        results (list of dict)
    """
    import pickle as pk
    from src import io

    with open(results, "rb") as fp:
        results = pk.load(fp)

    # load spec info from file
    spec_info = io.load_json(spec_file)
    
    return results, spec_info


def get_model_metrics(results, spec_info):
    """get model metrics for `results`. code has only been tested on results which have one classifier.

    Args: 
        results (list of dict): results output from `load_results`
        spec_file (str): full path to *.json spec file for each model. stored in `out-localspec-<modelname>`
    Returns:
        df_all (pd dataframe)
    """
    from src import io
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
        df['clf'] = res[0]['ml_wf.clf_info'][-1][1] # get classifier name (should always be the last list element in list)
        df = _add_model_parameters(df, spec_info=spec_info)

        df_all = pd.concat([df_all, df])
    
    return df_all


def order_across_splits(results, method='feature'):
    """get feature importances across splits

    Args:
        results (list of dict): 
    """
    import numpy as np  
    import pandas as pd

    df_features = pd.DataFrame()

    # extract importances (feature or permuation)
    if (method=='feature'):
        feature_splits = np.array(results.output.feature_importance)
    elif (method=='permutation'):
        feature_splits = np.array(results.output.permuation_importance)
    feature_names = np.array(results.output.feature_names)

    if len(feature_splits.shape) == 3:
        feature_splits = np.reshape(feature_splits, (feature_splits.shape[0], feature_splits.shape[2]))

    # rank features
    try:
        # get n splits and features
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
    except:
        pass

    return df_features


def model_based_importance(results):
    from sklearn.feature_selection import SelectFromModel
    import pandas as pd

    # get estimator steps and loop
    df_all = pd.DataFrame()
    estimator_steps = results.output.model.named_steps

    for name,estimator in estimator_steps.items():

        try:
            # get feature names
            feature_names = results.output.feature_names
            
            # get selector on prefit estimator
            selector = SelectFromModel(estimator=estimator, prefit=True)

            # get top features
            df = pd.DataFrame(data=selector.get_support(), columns=['top_features'])
            df['clf'] = name
            df['feature_names'] = feature_names
            df_all = pd.concat([df_all, df])
        except:
            continue
    
    return df_all


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
    df_features = pd.DataFrame(feature_importances, columns=['feature_names_common'])
    df_features['feature_probabilities_common'] = feature_probabilities
    
    return df_features 


def _sum_feature_weights(feature_splits, feature_names):
    """
    Calculate the sum of weights for each feature across splits.

    Parameters:
    - feature_splits (numpy.ndarray): An array of shape (n_splits, n_features) containing the weights for each feature across splits.
    - feature_names (list): A list of length n_features containing the names of the features.

    Returns:
    - df (pandas.DataFrame): A DataFrame with two columns: 'feature_sum' and 'feature_names_sum'. The 'feature_sum' column contains the summed weights for each feature, while the 'feature_names_sum' column contains the corresponding feature names.

    """
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


def save_to_existing_file(dataframe, fpath):
    import pandas as pd

    df = pd.DataFrame()
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
        except:
            pass
    df_out = pd.concat([df, dataframe])
    df_out.to_csv(fpath, index=False)



