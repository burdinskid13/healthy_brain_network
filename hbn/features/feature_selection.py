import warnings
warnings.filterwarnings("ignore")

def phenotype_features(
            target_spec,
            feature_spec,
            participants
            ):
    """make model features using whichever features are specified in "feature_spec" and whichever target specified in "target_spec"
    Features for selected "participants" are returned

    Args:
        target_spec (str): full path to target spec file 
        feature_spec (str): full path to feature spec file
        participants (list of str): list of fullpaths to participant files. Example ['../train_participants-ADHD.csv', '../train_participants-No_Diagnosis_Given.csv']
    Returns: 
        features_final (pd dataframe): features to be input to modeling routine
    """
    import pandas as pd
    from hbn import io
    from hbn.features import build_features

    # load from json file
    target_info = io.read_json(target_spec)
    feature_info = io.read_json(feature_spec)

    # make participants dataframe
    participants_df = pd.DataFrame()
    for participant in participants:
        participants_df = pd.concat([participants_df, pd.read_csv(participant)])

    # get features (X)
    features = build_features.get_features(
                assessment=feature_info['assessment'],
                domains=[feature_info['domains']],
                measures=[feature_info['measures']],
                min_num_participants=feature_info['min_num_participants']
                )

    # preprocess
    features_processed = build_features.preprocess(
                    dataframe=features,   
                    clf_info=feature_info['preprocessing'],
                    cols_to_ignore=['Identifiers']
                    )

    # combine features, targets, participants into one dataframe
    features_participants = features_processed.merge(participants_df, on='Identifiers')
    identifiers = features_participants['Identifiers'].tolist()
    targets = build_features.get_targets(
                            target_info=target_info, 
                            participants=identifiers
                            )

    features_final = features_participants.merge(targets, on='Identifiers').drop(['Identifiers'], axis=1)

    return features_final

def feature_selection_from_models():
    import pandas as pd

    df = pd.read_csv('classifier-feature_importance.csv')
    df[['feature_names_sum', 'feature_sum']]
