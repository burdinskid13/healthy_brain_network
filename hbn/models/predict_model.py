
def evaluation(fitted_model, X_test, y_test, feature_names):
    """perform evaluation on fitted model

    Args:   
        fitted_model (model): model object
        X_test (pd dataframe): test data
        y_test (pd series): test labels
        feature_names (list): feature names
    Returns:    
        df (pd dataframe): evaluation metrics
    """
    import pandas as pd
    from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score, r2_score

    # get predictions
    y_pred = fitted_model.predict(X_test)
    y_pred_prob = fitted_model.predict_proba(X_test)

    # get metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    auc = roc_auc_score(y_test, y_pred_prob[:, 1])
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # make eval dataframe
    data_dict = {'rmse': [rmse], 'auc': [auc], 'f1': [f1], 'precision': [precision], 'recall': [recall], 'r2': [r2]}
    df_eval = pd.DataFrame.from_dict(data_dict)

    # make predictions dataframe
    data_dict = {'y_test': y_test, 'y_pred': y_pred, 
                'y_pred_prob_0':y_pred_prob[:, 0], 'y_pred_prob_1':y_pred_prob[:, 1]}
    df_pred = pd.DataFrame.from_dict(data_dict)

    return df_eval, df_pred


def get_test_data(model_dir, model_spec): 
    """get test data from model directory

    Args:
        model_dir (str): fullpath to model directory
        model_spec (dict): dictionary of model spec
    Returns:
        X_test (pd dataframe): test data
        y_test (pd series): test labels
    """
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler

    
    # load test data
    test_data = pd.read_csv(os.path.join(model_dir, model_spec['filename']))

    # get target col
    target = model_spec['target_info']['target_column']

    # get test data
    y_test = np.array(test_data[target].tolist())

    # get X (features) columns
    x_cols = [col for col in test_data.columns if target not in col]
    X_test = test_data[x_cols]

    # figure out if x indices are lists of strings or numbers
    check_list = all(isinstance(x, str) for x in model_spec['x_indices'])

    # index test data using `x_indices` from model spec
    if check_list:
        X_test = X_test[model_spec['x_indices']]

    # standarize and scale? 
    clf = model_spec['clf_info'][0][0]
    if 'StandardScaler' in clf:
        scaler = StandardScaler()
        X_test_standarized = scaler.fit_transform(X_test)
        cols = X_test.columns
        X_test = pd.DataFrame(X_test_standarized, columns=cols)

    return X_test, y_test


def load_model(fpath): 
    """ load fitted model

    Args:
        fpath (str): fullpath to model
    Returns:
        model (model object)
    """
    import pickle as pkl

    with open(fpath, 'rb') as fp:
        results = pkl.load(fp)

    for res in results:
        if not res[0]['ml_wf.permute']:
            feature_names = res[1].output.feature_names
            fitted_model = res[1].output.model
    
    return fitted_model, feature_names
