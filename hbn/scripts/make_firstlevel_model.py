import warnings
warnings.filterwarnings("ignore")
import click
import os
import ast
import random
from pathlib import Path

from hbn import io
from hbn.features import build_features

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def load_specs(feature_spec, target_spec, participant_spec):
    """ Load in `feature_spec`, `target_spec`, and `participant_spec` 

    Args: 
        feature_spec (str): full path to feature spec
        target_spec (str): full path to target spec
        participant_spec (str): full path to participant spec
    Returns:
        feature_info (dict), target_info (dict), participant_info (dict)
    """
    # load spec files
    feature_info = io.load_json(feature_spec)
    target_info = io.load_json(target_spec)
    participant_info = io.load_json(participant_spec)

    return feature_info, target_info, participant_info


def get_data(feature_info, target_info, participant_info, dirn):
    """ Get features, targets, and participant dataframes using parametesr from `feature_spec`, `target_spec`, and `participant_spec`

    Args: 
        feature_info (dict): dictionary loaded from `feature_spec`
        target_info (dict): dictionary loaded from `target_spec`
        participant_info (dict): dictionary loaded from `participant_spec`
        dirn (str): directory where data files are stored
    Returns (pd.DataFrame):
        df_features, df_target, df_participants
    """
    df_features = build_features.get_features(feature_info, dirn) 
    df_target = build_features.get_targets(target_info, dirn)
    df_participants = build_features.get_participants(participant_info, dirn)

    return df_features, df_target, df_participants


def check_participant_id(participant_id, df_features, df_target, df_participants):
    """check that `participant_id` column is in `df_features`, `df_target`, and `df_participants`

    Args: 
        participant_id (str): the column name of the participant identifier
        df_features (pd.DataFrame): dataframe of features
        df_target (pd.DataFrame): dataframe of targets
        df_participants (pd.DataFrame): dataframe of participants
    Returns: 
        check_id (bool): True if `participant_id` is in `df_features`, `df_target`, and `df_participants`
    """

    check_id = (participant_id in df_features.columns) and (participant_id in df_target.columns) and (participant_id in df_participants.columns)

    if not check_id:
        raise ValueError(f'{participant_id} is not in `df_features`, `df_target`, and `df_participants`')
    else:
        return check_id

    
def chain_dicts(dicts):
    """
    Chains together multiple dictionaries into a single dictionary.

    Args:
        dicts: A list of dictionaries to chain together.

    Returns:
        A single dictionary that contains the key-value pairs from all of the dictionaries in the input list.
    """

    chained_dict = {}
    for d in dicts:
        chained_dict.update(d)
    return chained_dict


def train_test_split(features, info): 
    """split `features` into train and test sets based on info given in `info`
    Args:
        features (pd.DataFrame): dataframe of features
        info (dict): dictionary of info for splitting. dict loaded from `../model_specs/participant-<name>-spec.json`
    Returns:
        df_train (pd.DataFrame), df_test (pd.DataFrame)
    """
    split_col = info['train_test_split']['split_col']
    train = info['train_test_split']['train']
    test = info['train_test_split']['test']

    df_train = features[features[split_col]==train]
    df_test = features[features[split_col]==test]

    df_train_drop = df_train.drop(split_col, axis=1).reset_index(drop=True)
    df_test_drop = df_test.drop(split_col, axis=1).reset_index(drop=True)

    return df_train_drop, df_test_drop


def make_features(
    feature_info,
    target_info,
    participant_info,
    data_dir
    ):
    """Make model to be input to pydra-ml using the following: `feature_info`, `target_info`, `participant_info`
    Saves model spec to `out_dir`

    Args:
        feature_info (dict): dictionary loaded from `feature_spec`
        target_info (dict): dictionary loaded from `target_spec`
        participant_info (dict): dictionary loaded from `participant_spec`
        data_dir (str): directory where `filename` stored in `feature_spec`, `target_spec`, and `participant_spec` are saved. these files should all be saved in the same directory. 
    """

    # get features, targets, and participants dataframes
    df_features, df_target, df_participants = get_data(feature_info, target_info, participant_info, dirn=data_dir)
    print(f'loaded data from {data_dir}', flush=True)

    # get participant id from `participant_spec` - this column will be ignored in the preprocessing routine
    participant_id = participant_info['participant_id']
    
    # check if `participant_id` is present in all dataframes (raises error if not)
    check_participant_id(participant_id, df_features, df_target, df_participants)

    # combine features and targets
    combined_df = build_features.combine_features_and_targets(
        features=df_features, 
        targets=df_target,
        merge_on=participant_id,
        )
    print(f'combined features and targets', flush=True)

    # filter participants
    filter_cols = [participant_id, target_info['target_column'], participant_info['train_test_split']['split_col']]
    merge_cols = [participant_id, target_info['target_column']]

    df_merged = build_features.merge_with_participants(
        dataframe=combined_df, 
        participants=df_participants[filter_cols], 
        participant_id=participant_id, 
        merge_cols=merge_cols
        )
    print(f'filtered participants', flush=True)

    # drop duplicates 
    df_merged = df_merged.drop_duplicates().dropna(how='all', axis=1)

    # preprocess combined dataframe and drop participant id
    features_preprocessed = build_features.preprocess(
                    dataframe=df_merged,  
                    clf_info=feature_info['clf_info'],
                    cols_to_ignore=filter_cols, 
                    cols_to_drop=feature_info['cols_to_drop'],
                    threshold=feature_info['threshold'],
                    target_column=target_info['target_column'],
                    binarize_target=target_info['binarize']
                    )

    # split into train/test
    df_train, df_test = train_test_split(features=features_preprocessed.drop(participant_id, axis=1), info=participant_info)

    # upsample training data (leave test intact)
    target_col = target_info['target_column']
    x_indices = [col for col in df_train.columns if target_col not in col]
    df_train = build_features.upsample_data(y_train=df_train[[target_col]], X_train=df_train[x_indices])

    return df_train, df_test, x_indices, [target_col], features_preprocessed[filter_cols].reset_index(drop=True)


def make_model_spec(
    x_indices, 
    target_vars,
    pydraml_spec,
    filename
    ):
    """Make model spec to be input to pydra-ml using the following: `feature_spec`, `target_spec`, `participant_spec`, `pydraml_spec`

    Args:
        x_indices (list of str): list of feature names.
        target_vars (list of str): list of target names.
        pydraml_spec (str): fullpath to pydra-ml spec.
        filename (str): name of features file.
    Returns:
        pydraml_info (dict):
    """
    # load pydra-ml spec
    pydraml_info = io.load_json(pydraml_spec)

    # update pydraml info
    pydraml_info['filename'] = filename
    pydraml_info['x_indices'] =  x_indices
    pydraml_info['target_vars'] = target_vars

    return pydraml_info


def run(
    feature_spec,
    target_spec,
    participant_spec,
    pydraml_spec,
    data_dir,
    out_dir
    ):
    """ run predictive models using pydra-ml. Model features and model spec are saved to `out_dir`. 

    Args:
        feature_spec (str): full path to feature spec file.
        target_spec (str): full path to target spec file.
        participant_spec (str): full path to participant spec file.
        pydraml_spec (str): full path to pydra-ml spec file.
        data_dir (str): directory where `filename` in `feature_spec`, `target_spec`, and `participant_spec` are saved. These files should all be saved in the same directory.
        out_dir (str): directory where model features and spec should be saved.
    """

    # load parameters from spec files
    feature_info, target_info, participant_info = load_specs(feature_spec, target_spec, participant_spec)

    # get features
    df_train, df_test, x_indices, target_vars, df_index = make_features(feature_info,   
                                                                        target_info,
                                                                        participant_info,
                                                                        data_dir
                                                                        )
    
    # name of train and test
    train = participant_info['train_test_split']['train']
    test = participant_info['train_test_split']['test']

    # only save out features + spec if not empty
    for (features, fname) in zip([df_train, df_test], [train, test]):

        # get model name
        filename = f'features-{fname}.csv'

        # get model spec
        model_info = make_model_spec(
                                    x_indices, 
                                    target_vars,
                                    pydraml_spec,
                                    filename=filename
                                    )
        
        # update model info with features, targets, participants
        feature_info['spec_name'] = Path(feature_spec).stem
        feature_info['model_type'] = 'firstlevel'
        target_info['spec_name'] = Path(target_spec).stem
        participant_info['spec_name'] = Path(participant_spec).stem
        model_info_updated = chain_dicts([model_info, 
                                          {'feature_info': feature_info}, 
                                          {'target_info': target_info}, 
                                          {'participant_info': participant_info}]
                                          )

        # save out model features and spec 
        io.make_dirs(out_dir) # create directory if it doesn't exist 
        feature_path = os.path.join(out_dir, filename)
        features.to_csv(feature_path, index=False)
        spec_path = os.path.join(out_dir, f'model_spec-{fname}.json')
        io.save_json(spec_path, model_info_updated)
        print(f'created new file: {filename} and model spec file: model_spec-{fname}.json in {out_dir}')
    
    # save out index for train and test
    df_index.to_csv(os.path.join(out_dir, 'participant_index.csv'), index=False)

    # return specs for training data
    feature_path = os.path.join(out_dir, f'features-{train}.csv')
    spec_path = os.path.join(out_dir, f'model_spec-{train}.json')
    
    return feature_path, spec_path


if __name__ == "__main__":
    run()