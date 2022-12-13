import os
from hbn.constants import Defaults
    

def make_model(
    feature_spec,
    target_spec,
    pydraml_spec,
    participants,
    out_dir=Defaults.MODEL_SPEC_DIR
    ):
    """make model spec file using the following:`feature specs`, `targets`, `participants`, `pydraml_spec`  
    
    Models are only created if they satisfy the following conditions:
    more than one feature, more than one target class, more than 100 participants

    Args: 
        feature_spec (str): fullpath to feature spec
        target_spec (str): fullpath to target spec
        pydraml_spec (str): fullpath to pydraml spec
        participants (list of str): For example: ['../train_participants-ADHD.csv', '../train_participants-No_Diagnosis_Given.csv']
        out_dir (str): directory where model spec and feature file should be saved
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
    
    # set spec + features filenames
    random_number = round(random.random()*1000000000)
    filename = f'model_features_{random_number}.csv'
    print(f'trying to make new filename: {filename}')

    # get participant identifiers
    participants_all = pd.DataFrame()
    for participant in participants:
        participants_all = pd.concat([participants_all, pd.read_csv(participant)])

    model_spec = None; model_features = None
    try:
        # make multiple model specs using features, target, and participant specs 
        dataframe = feature_selection.phenotype_features(
                            feature_spec=feature_spec, 
                            target_spec=target_spec,
                            participants=participants
                            )

        # set certain conditionals for model spec to be run and model features to be created
        # there have to be more than one column, more than one unique target, more than 100 participants
        conditionals = all((dataframe.shape[1]>1, len(dataframe[target_info['outname']].unique())>1, dataframe.shape[0]>100))
        
        if conditionals: 

            # chain together dictionaries
            pydraml_info = io.read_json(pydraml_spec)
            pydraml_info.update({'target_spec': target_info})
            pydraml_info.update({'feature_spec': feature_info})
            pydraml_info.update({'participants': participants_all['Identifiers'].tolist()}) 

            # update model spec with features filename
            model_features = os.path.join(out_dir, filename)
            pydraml_info['filename'] = Path(model_features).name
            pydraml_info['x_indices'] =  [*range(1,len(dataframe.columns)-1)]
            pydraml_info['target_vars'] = target_info['outname']

            # get model spec name
            spec_name = 'classifier-' + '_'.join(re.split(r'_|,|/| ', feature_info['measures'])) + '-' + target_info['outname'] + '-spec.json'
            model_spec = os.path.join(out_dir, spec_name)
            io.save_dict_as_JSON(model_spec, pydraml_info)

            # save out model features
            dataframe.to_csv(model_features, index=False)

        else:
            print(f'model spec not created for {filename} because one of the following conditions was not met: more than 1 feature, more than one unique target, more than 100 participants')
    except:
       print(f'failed to make model specs for {filename}')

    return model_spec, model_features


def run_pydra_ml(
    model_spec, 
    spec_dir,
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/',
    out_dir=Defaults.MODEL_DIR):
    """ run predictive models using pydra-ml. must provide `model_spec` json and `filename` in `model_spec` must be a csv of features saved in ../features/

    Args:
        model_spec (str): full path to model spec file
        spec_dir (str): model spec directory (where `filename` in model_spec is temporarily stored)
        cachedir (str): default is '/Users/maedbhking/pydra-ml/cache-wf/'
        out_dir (str): full path to model output directory
    Returns: 
        saves (pickled) model to ../data/interim/
    """
    # load libraries
    import os
    import pandas as pd
    import glob
    import shutil
    from hbn import io
    from pydra_ml.classifier import gen_workflow, run_workflow

    # create cachedir if it hasn't already been created
    io.make_dirs(cachedir)
    io.make_dirs(out_dir)

    # load model spec json
    spec_info = io.read_json(model_spec)

    print(f'running {model_spec}...\n')
    print("spec info", spec_info)
    
    filename = os.path.join(spec_dir, spec_info['filename'])
    spec_info['filename'] = filename # full path to csv file
    wf = gen_workflow(spec_info, cache_dir=cachedir)
    run_workflow(wf, "cf", {"n_procs": 1})

    # move model output to new directory + add model spec file
    out_models = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
    shutil.move(model_spec, out_models[0])
    shutil.move(filename, out_models[0])
    shutil.move(out_models[0], out_dir)


def secondlevel_summary(
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
    from hbn import io

    # make model spec if it doesn't already exist
    io.make_dirs(out_dir)

    # get results file
    model_name = Path(results_dir).name.split('-')[2]
    results_file = os.path.join(results_dir, f'results-{model_name}.pkl')

    # get model spec file
    spec_file = glob.glob(os.path.join(results_dir, '*.json'))[0]

    # load results
    results, spec_info = load_results(results=results_file, spec_file=spec_file)
    clf = Path(spec_file).name.split('-')[0] # 'classifier' or 'regression'

    # loop over results and get feature and permuation importances
    for res in results:
        if not res[0]['ml_wf.permute']:
            # only if data are not permuted
            # get feature importances
            df_features = get_feature_importance(model_name, results=res, spec_info=spec_info)
            feature_fname = f'{clf}-feature_importance.csv'
            _save_to_existing_file(dataframe=df_features, fpath=os.path.join(out_dir, feature_fname))

            # get permuation importances
            df_permutation = get_permutation_importance(model_name, results=res, spec_info=spec_info)
            permutation_fname = f'{clf}-permutation_importance.csv'
            _save_to_existing_file(dataframe=df_permutation, fpath=os.path.join(out_dir, permutation_fname))

    # get model summary (and save to disk)
    model_dataframe = make_model_summary(
                    model_name,
                    results=results, 
                    spec_info=spec_info, 
                    )
    model_fname = f'{clf}-all-phenotypic-models-performance.csv'
    _save_to_existing_file(dataframe=model_dataframe, fpath=os.path.join(out_dir, model_fname))


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


def _add_model_parameters(dataframe, model_name, spec_info, results):
    """add model parameters to dataframe
    """
    # add spec info 
    features = spec_info['feature_spec']['assessment'] + '-' + spec_info['feature_spec']['domains'] +'-' + spec_info['feature_spec']['measures']

    dataframe['participants'] = '-'.join(spec_info['participants'])
    dataframe['model'] = model_name
    dataframe['clf'] = results['ml_wf.clf_info'][1]
    dataframe['target'] = spec_info['target_vars'][0]
    dataframe['features'] = features
    dataframe['assessment'] = spec_info['feature_spec']['assessment']
    dataframe['domains'] = spec_info['feature_spec']['domains']
    dataframe['measures'] = spec_info['feature_spec']['measures']

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

        # add model parameters
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

