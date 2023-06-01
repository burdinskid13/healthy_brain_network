import os
from hbn.constants import Defaults
from hbn.data import make_dataset
    

def make_model(
    feature_spec,
    target_spec,
    pydraml_spec,
    participant_spec,
    drop_identifiers=True,
    out_dir=Defaults.MODEL_SPEC_DIR
    ):
    """make model spec file using the following:`feature specs`, `targets`, `participant_spec`, `pydraml_spec`  
    
    Models are only created if they satisfy the following conditions:
    more than one feature, more than one target class, more than 100 participants

    Args: 
        feature_spec (str): fullpath to feature spec
        target_spec (str): fullpath to target spec
        pydraml_spec (str): fullpath to pydraml spec
        participant_spec (str): fullpath to participant spec
        out_dir (str): directory where model specs should be saved
    Returns:
        model_spec (str): full path to model spec
    """
    import re
    import os
    from hbn import io
    import pandas as pd
    import random
    from pathlib import Path
    from hbn.features import feature_selection

    # load in spec files
    target_info = io.read_json(target_spec)
    feature_info = io.read_json(feature_spec)
    participant_info = io.read_json(participant_spec)
    
    # set spec + features filenames
    random_number = round(random.random()*1000000000)
    filename = f'model_features_{random_number}.csv'
    io.make_dirs(out_dir) # make directory if it doesn't already exist

    # get participant identifiers from spec
    participants_all = make_dataset.get_participants(split=participant_info['split'], 
                                                    disorders=participant_info['diagnoses'], 
                                                    age=participant_info['age'],
                                                    sex=participant_info['sex']
                                                    )

    model_spec = None; model_features = None
    # make multiple model specs using features, target, and participant specs 
    df_features = feature_selection.phenotype_features(
                        feature_spec=feature_spec, 
                        target_spec=target_spec,
                        participants=participants_all,
                        drop_identifiers=drop_identifiers,
                        )

    # set certain conditionals for model spec to be run and model features to be created
    # there have to be more than one column, more than one unique target, more than 50 participants
    conditionals = all((df_features.shape[1]>1, len(df_features[target_info['outname']].unique())>1, df_features.shape[0]>50))
    
    if conditionals: 
        
        # chain together dictionaries
        pydraml_info = io.read_json(pydraml_spec)
        pydraml_info.update({'target_spec': target_info})
        pydraml_info.update({'feature_spec': feature_info})
        pydraml_info.update({'participant_spec': participant_info})
        pydraml_info.update({'participants': participants_all}) 

        # update model spec with features filename
        model_features = os.path.join(out_dir, filename)
        pydraml_info['filename'] = Path(model_features).name
        pydraml_info['x_indices'] =  [*range(1,len(df_features.columns)-1)]
        pydraml_info['target_vars'] = target_info['outname']

        # get model spec name
        spec_name = 'classifier-' + '_'.join(re.split(r'_|,|/| ', feature_info['assessment'])) + '-' + '_'.join(re.split(r'_|,|/| ', feature_info['measures'])) + '-' + feature_info['abbrevs'] + '-' + target_info['outname']
        model_spec = os.path.join(out_dir, spec_name + '-spec.json')
        if os.path.isfile(model_spec):
            spec_name =  spec_name + '-' + str(round(random.random()*1000000000)) + '-spec.json'
            model_spec = os.path.join(out_dir, spec_name)

        # save out model features and spec
        if not df_features.empty:
            df_features.to_csv(model_features, index=False)
            io.save_dict_as_JSON(model_spec, pydraml_info)

            print(f'created new file: {filename} and model spec file: {spec_name} in {out_dir}')

    else:
        print(f'model spec not created for {filename} because one of the following conditions was not met: more than 1 feature, more than one unique target, more than 100 participants')

    return model_spec, model_features


def run_pydra_ml(
    model_spec, 
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/',
    out_dir=Defaults.MODEL_DIR):
    """ run predictive models using pydra-ml. must provide `model_spec` json and `filename` in `model_spec` must be a csv of features saved in ../features/

    Args:
        model_spec (str): full path to model spec file
        cachedir (str): default is '/Users/maedbhking/pydra-ml/cache-wf/'
        out_dir (str): full path to model output directory
    Returns: 
        saves (pickled) model to ../data/interim/
    """
    # load libraries
    import os
    from hbn import io
    from pathlib import Path
    from pydra_ml.classifier import gen_workflow, run_workflow

    # create cachedir if it hasn't already been created
    io.make_dirs(cachedir)
    io.make_dirs(out_dir)

    # load model spec json
    spec_info = io.read_json(model_spec)

    # get directory where `model_spec` is stored
    spec_dir = Path(model_spec).parent

    print(f'running {model_spec}...\n', flush=True)
    print("spec info", spec_info, flush=True)

    spec_info['filename'] = os.path.join(spec_dir, spec_info['filename']) # full path to csv file

    # change directory to output directory
    os.chdir(out_dir)
    print(f'changing directory to {out_dir}')

    # run workflow
    wf = gen_workflow(spec_info, cache_dir=cachedir)
    run_workflow(wf, "cf", {"n_procs": 1})


def evaluation(results_dir, test_spec):
    import os
    import glob
    import pickle as pk
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import mean_squared_error
    from hbn import io
    from hbn.constants import Defaults

    """Args:
        results_dir (str): fullpath to top-level results dir. for example '../out-localspec-<>'
    """

    print(f"calculating evaluation for {results_dir}")

    # get results file
    model_name = Path(results_dir).name.split('-')[2]
    fitted_model = os.path.join(results_dir, f'results-{model_name}.pkl')

    # get fitted model
    with open(fitted_model, "rb") as fp:
        results = pk.load(fp)

    # loop over results and get model + feature names (only if data are not permuted)
    df = pd.DataFrame()
    for res in results:
        if not res[0]['ml_wf.permute']:
            feature_names = res[1].output.feature_names
            fitted_model = res[1].output.model

            # get model spec info for fitted model
            fpath = glob.glob(os.path.join(Path(results_dir).parent, '*-spec.json*'))[0]
            feature_spec = io.read_json(fpath)['feature_spec']
            target_spec = io.read_json(fpath)['target_spec']
            test_spec = io.read_json(os.path.join(Defaults.MODEL_SPEC_DIR, test_spec))

            # get test data
            X, y = get_test(feature_spec=feature_spec, target_spec=target_spec, test_spec=test_spec, feature_names=feature_names)

            # get predictions 
            y_pred = fitted_model.predict(X)

            # get rmse
            rmse = mean_squared_error(y, y_pred, squared=False)
            R = calculate_R(y, y_pred)
            R2 = calculate_R2(y, y_pred)

            # make dataframe
            df['y'] = y
            df['y_pred'] = y_pred
            df['train'] = io.read_json(fpath)['participant_spec']['diagnoses'][0]
            df['predict'] = test_spec['diagnoses'][0]
            df['split'] = test_spec['split']
            df['R2'] = R2
            df['R'] = R
            df['rmse'] = rmse

    return df


def get_test(feature_spec, target_spec, test_spec, feature_names):
    import os
    import pandas as pd
    from hbn import io
    from hbn.constants import Defaults
    from hbn.data.make_dataset import get_participants
    from hbn.features.feature_selection import phenotype_features   

    # get participant identifiers (test)
    participants = get_participants(split='train', 
                                    disorders=test_spec['diagnoses'], 
                                    age=test_spec['age'],
                                    sex=test_spec['sex']
                                    )

    # get test data using feature spec from fitted model + participant_spec
    features = phenotype_features(feature_spec=feature_spec,
                                participants=participants,
                                target_spec=target_spec,
                                drop_identifiers=True
                                )

    # make new dataframe (with same columns as feature names)
    X = pd.DataFrame(columns=feature_names)
    for col in X:
        if col in features.columns:
            X[col] = features[col]
        else:
            X[col] = 0
    
    y = features[target_spec['outname']]

    return X, y


def calculate_R(y, y_pred):
    """Calculates correlation between Y and Y_pred without subtracting the mean.
    Args:
        Y (nd-array):
        Y_pred (nd-array):
    Returns:
        R (scalar): Correlation between Y and Y_pred
    """
    import numpy as np

    SYP = np.nansum(y * y_pred, axis=0)
    SPP = np.nansum(y_pred * y_pred, axis=0)
    SST = np.sum(y ** 2, axis=0)  # use np.nanmean(Y) here?

    R = np.nansum(SYP) / np.sqrt(np.nansum(SST) * np.nansum(SPP))

    return R


def calculate_R2(y, y_pred):
    """Calculates squared correlation between Y and Y_pred without subtracting the mean.
    Args:
        Y (nd-array):
        Y_pred (nd-array):
    Returns:
        R2 (scalar): Squared Correlation between Y and Y_pred
    """
    import numpy as np

    res = y - y_pred

    SSR = np.nansum(
        res ** 2, axis=0
    )  # remember: without setting the axis, it just "flats" out the whole array and sum over all
    SST = np.sum(y ** 2, axis=0)  # use np.nanmean(Y) here??

    R2 = 1 - (np.nansum(SSR) / np.nansum(SST))

    return R2


def secondlevel_summary(
    results,
    spec,
    out_dir=Defaults.MODEL_DIR,
    methods=['feature'] # 'permuation'
    ):
    """Makes model and feature summary files from results output from `run_model_pipeline_firstlevel`

    Saves output in `../interim/models/`

    Args:
        results (str): fullpath to results file (.pkl)
        spec (str): fullpath to spec file (.json)
        out_dir (str): directory where second level modeling summary will be saved
        methods (list of str): feature interpretability based on feature or permuation importances. default is ['feature']
    """
    import glob
    import os
    from pathlib import Path
    from hbn import io

    # make model out_dir if it doesn't already exist
    io.make_dirs(out_dir)

    # load results
    data, spec_info = load_results(results=results, spec_file=spec)
    clf_name = Path(spec).name.split('-')[0] # 'classifier' or 'regression'
    model_name = Path(results).stem.split('-')[1]

    # loop over results and get feature and permuation importances
    for res in data:
        # only if data are not permuted
        if not res[0]['ml_wf.permute']:
            for method in methods:
                df = feature_interpretability(results=res[1], spec_info=spec_info, method=method)
                df['model'] = model_name
                _save_to_existing_file(dataframe=df, fpath=os.path.join(out_dir, f'{clf_name}-{method}_importance.csv'))

    # get model summary (and save to disk)
    model_dataframe = make_model_summary(results=data, spec_info=spec_info)
    model_dataframe['model'] = model_name
    model_fname = f'{clf_name}-all-phenotypic-models-performance.csv'
    _save_to_existing_file(dataframe=model_dataframe, fpath=os.path.join(out_dir, model_fname))


def feature_interpretability(results, spec_info, method='feature'):
    import pandas as pd

    df1 = order_across_splits(results=results, method=method)
    df2 = model_based_importance(results=results)

    # concat into features dataframe
    df_features = pd.concat([df1, df2], axis=1)

    # add model parameters
    df_features = _add_model_parameters(df_features, spec_info=spec_info)

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
    from hbn import io

    with open(results, "rb") as fp:
        results = pk.load(fp)

    # load spec info from file
    spec_info = io.read_json(spec_file)
    
    return results, spec_info


def check_models(dirn=Defaults.MODEL_DIR,
                diagnosis_file=None,
                filter='*2023*'):
    import glob
    import pandas as pd
    from pathlib import Path
    from hbn.constants import Defaults
    from hbn.data.make_dataset import make_summary

    models = glob.glob(os.path.join(dirn, filter))

    # load in clinical diagnosis
    if diagnosis_file is None:
        dx = make_summary(save=False)
    else:
        dx = pd.read_csv(diagnosis_file)

    model_name =  'classifier-all-phenotypic-models-performance.csv'
    feature_name = 'classifier-feature_importance.csv'

    df_all = pd.DataFrame()
    # loop over models
    for model_dir in models:
        try:
            # load models
            df = pd.read_csv(os.path.join(model_dir, model_name))
            df_feature = pd.read_csv(os.path.join(model_dir, feature_name))

            # make participants dataframe
            participants = df['participants'].loc[0].split("-")
            df_part = pd.DataFrame(participants, columns=['Identifiers'])

            # get target
            target = df['target'].unique().tolist()

            # merge participants with diagnosis and sex
            diagnoses = dx.merge(df_part, on=['Identifiers'])['DX_01'].unique().tolist()
            category = dx.merge(df_part, on=['Identifiers'])['DX_01_Cat_new'].unique().tolist()
            sex = dx.merge(df_part, on=['Identifiers'])['Sex'].unique().tolist()
            age = dx.merge(df_part, on=['Identifiers'])['Age'].round().unique().astype(int)
    
            # make into str
            if len(age)>1:
                min_age = min(age); max_age = max(age)
                age = f'{min_age:02d}-{max_age:02d}'
            else:
                age = f'{age[0]:02d}'
            
            if len(sex)>1:
                sex = 'all'
            else:
                sex = sex[0]
            
            # add new columns to dataframe
            df['diagnoses'], df['category'], df['sex'], df['age'] = '_'.join(diagnoses), '_'.join(category), sex, age
            df['data'] = df['data'].map({'model-data': 'null', 'model-null': 'data'})
            #df['top_features'] = 
            
            # get model name
            model_name = Path(model_dir).name
            
            # print out models
            #print(f'{model_name}: {diagnoses}: {target}: {sex}: {age}')
            
            # concat dataframes
            df_all = pd.concat([df_all, df])

        except:
            pass
            print(f'{fname} does not exist for {model_dir}, run `run_second_level.sh`')
        
    return df_all


def _add_model_parameters(dataframe, spec_info):
    """add model parameters from spec_info to dataframe
    """
    # add spec info 
    features = spec_info['feature_spec']['assessment'] + '-' + spec_info['feature_spec']['abbrevs']

    dataframe['participants'] = '-'.join(spec_info['participants'])
    try:
        dataframe['target'] = spec_info['target_vars']
    except:
        dataframe['target'] = spec_info['target_vars'][0]
    dataframe['features'] = features
    dataframe['assessment'] = spec_info['feature_spec']['assessment']
    dataframe['domains'] = spec_info['feature_spec']['domains']
    dataframe['measures'] = spec_info['feature_spec']['measures']
    dataframe['abbrevs'] = spec_info['feature_spec']['abbrevs']

    return dataframe


def make_model_summary(results, spec_info):
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
        df['clf'] = res[0]['ml_wf.clf_info'][1]
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
    if method=='feature':
        feature_splits = np.array(results.output.feature_importance)
    elif method=='permutation':
        feature_splits = np.array(results.output.permuation_importance)
    feature_names = np.array(results.output.feature_names)

    if len(feature_splits.shape) == 3:
        feature_splits = np.reshape(feature_splits, (feature_splits.shape[0], feature_splits.shape[2]))
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

