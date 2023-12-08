import warnings
warnings.filterwarnings("ignore")
import click
import os
import ast
import random

from src import io
from src.features import build_features

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def _load_specs(feature_spec, target_spec, participant_spec):
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


def _get_data(feature_info, target_info, participant_info, dirn):
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


def _check_participant_id(participant_id, df_features, df_target, df_participants):
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

    
def _chain_dicts(dicts):
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
    df_features, df_target, df_participants = _get_data(feature_info, target_info, participant_info, dirn=data_dir)

    # get participant id from `participant_spec` - this column will be ignored in the preprocessing routine
    participant_id = participant_info['participant_id']
    
    # check if `participant_id` is present in all dataframes (raises error if not)
    _check_participant_id(participant_id, df_features, df_target, df_participants)

    # combine features and targets, and filter dataframe by participants
    df_combined = build_features.combine_features_and_targets(
        features=df_features, 
        targets=df_target,
        participant_id=participant_id,
        participants=df_participants,
        )

    # columns we want to ignore in the preprocessing routines: participant id, target
    cols_to_ignore = [participant_id, target_info['target_column']]
    
    # preprocess combined dataframe and drop participant id
    features_preprocessed = build_features.preprocess(
                    dataframe=df_combined,  
                    clf_info=feature_info['clf_info'],
                    cols_to_ignore=cols_to_ignore,
                    cols_to_drop=feature_info['cols_to_drop'],
                    threshold=feature_info['threshold'],
                    target_column=target_info['target_column']
                    ).drop(participant_id, axis=1)

    # get x indices (all features except target) and target vars
    x_indices = [i for i, string in enumerate(features_preprocessed.columns) if string != target_info['target_column']]
    target_vars = target_info['target_column']

    return features_preprocessed, x_indices, target_vars


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
    feature_info, target_info, participant_info = _load_specs(feature_spec, target_spec, participant_spec)

    # get features
    features, x_indices, target_vars = make_features(feature_info,
                                                    target_info,
                                                    participant_info,
                                                    data_dir
                                                    )                                

    # only save out features + spec if not empty
    if not features.empty:

        # get model name
        randm = random.randint(10000,1000000)
        filename = f'features-{randm}.csv'

        # get model spec
        model_info = make_model_spec(
                                    x_indices, 
                                    target_vars,
                                    pydraml_spec,
                                    filename=filename
                                    )
        
        # update model info with features, targets, participants
        model_info_updated = _chain_dicts([model_info, {'feature_info': feature_info}, {'target_info': target_info}, {'participant_info': participant_info}])

        # save out model features and spec 
        io.make_dirs(out_dir) # create directory if it doesn't exist 
        features.to_csv(os.path.join(out_dir, filename), index=False)
        io.save_json(os.path.join(out_dir, f'model_spec-{randm}.json'), model_info_updated)
        print(f'created new file: {filename} and model spec file: model_spec-{randm}.json in {out_dir}')


if __name__ == "__main__":
    run()