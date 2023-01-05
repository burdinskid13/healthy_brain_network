import warnings
warnings.filterwarnings("ignore")

def phenotype_features(
            feature_spec,
            participants,
            target_spec=None,
            preprocess=True,
            drop_identifiers=True,
            ):
    """make model features using whichever features are specified in "feature_spec" and whichever target specified in "target_spec"
    Features for selected "participants" are returned

    Args: 
        feature_spec (str): full path to feature spec file
        participants (list of str or pd.DataFrame): list of fullpaths to participant files OR pd Dataframe with 'Identifiers' column indicating participants. Example ['../train_participants-ADHD.csv', '../train_participants-No_Diagnosis_Given.csv']
        target_spec (str or None): full path to target spec file. if None, then only features are returned.
        preprocess (bool): default is True.
        drop_identifiers (bool): default is True (returns dataframe without 'Identifiers' column)
    Returns: 
        features_final (pd dataframe): features to be input to modeling routine
    """
    import pandas as pd
    from hbn import io
    from hbn.features import build_features

    # load from json file
    feature_info = io.read_json(feature_spec)

    # make participants dataframe if list of csv files is given as input
    if isinstance(participants, list):
        participants_list = participants
        participants = pd.DataFrame()
        for participant in participants_list:
            participants = pd.concat([participants, pd.read_csv(participant)])

    # get features (X)
    features = build_features.get_features(
                assessment=feature_info['assessment'],
                domains=[feature_info['domains']],
                measures=[feature_info['measures']],
                min_num_participants=feature_info['min_num_participants']
                )

    # analyze missing data
    #features.to_csv('/om2/user/shreyark/healthy_brain_network/hbn/features/features_preprocessed.csv')
    # preprocess
    if preprocess:
        features = build_features.preprocess(
                        dataframe=features,   
                        clf_info=feature_info['preprocessing'],
                        cols_to_ignore=['Identifiers']
                        )

    # combine features, targets, participants into one dataframe
    features_participants = features.merge(participants, on='Identifiers')
    identifiers = features_participants['Identifiers'].tolist()

    targets = pd.DataFrame(identifiers, columns=['Identifiers'])
    if target_spec is not None:
        target_info = io.read_json(target_spec)
        targets = build_features.get_targets(
                                target_info=target_info, 
                                participants=identifiers
                                )   
    if drop_identifiers:
        features_final = features_participants.merge(targets, on='Identifiers').drop(['Identifiers'], axis=1)
    else:
        features_final = features_participants.merge(targets, on='Identifiers')
        
    # upsample minority class using smote 
    features_final = build_features.smote(features_final)
    
    return features_final


def feature_selection_from_models():
    import pandas as pd

    df = pd.read_csv('classifier-feature_importance.csv')
    df[['feature_names_sum', 'feature_sum']]
