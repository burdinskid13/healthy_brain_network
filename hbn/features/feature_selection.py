import warnings
warnings.filterwarnings("ignore")

def phenotype_features(
            feature_spec,
            participants,
            target_spec=None,
            preprocess=True,
            drop_identifiers=True,
            oversample=False
            ):
    """make model features using whichever features are specified in "feature_spec" and whichever target specified in "target_spec"
    Features for selected "participants" are returned

    Args: 
        feature_spec (str or dict): full path to feature spec file OR dict loaded from file
        participants (list of str): list of participant identifiers (output from `make_dataset.get_participants`)
        target_spec (str or None): (optional) full path to target spec file. if None, then only features are returned.
        preprocess (bool): (optional) default is True.
        drop_identifiers (bool): (optional) default is True (returns dataframe without 'Identifiers' column)
        oversample (bool): (optional) oversample minority class of dataframe

    Returns: 
        features_final (pd dataframe): features to be input to modeling routine
    """
    import pandas as pd
    from hbn import io
    import os
    from hbn.constants import Defaults
    from hbn.features import build_features

    # load from json file
    if isinstance(feature_spec, str):
        feature_spec = io.read_json(feature_spec)

    # make participants dataframe 
    participants_df = pd.DataFrame(participants, columns=['Identifiers'])

    # get features (X)
    features = build_features.get_features(
                assessment=feature_spec['assessment'],
                domains=[feature_spec['domains']],
                measures=[feature_spec['abbrevs']]
                )
    
    # filter based on participants
    features = features.merge(participants_df, on='Identifiers')
    
    # optionally add demographics as features
    if feature_spec['add_demos'] is not None:
        demos_df = pd.read_csv(os.path.join(Defaults.FEATURE_DIR, feature_spec['add_demos']['filename']))
        tmp = demos_df.merge(features, on=['Identifiers'])
        cols_to_factorize = feature_spec['add_demos']['features']
        for col in cols_to_factorize:
            if col in tmp.columns:
                tmp.loc[:, col] = tmp[col].factorize()[0]
        features = tmp

    # remove `features_to_ignore` from dataframe if any are provided in `target_spec`
    if target_spec is not None:
        target_info = io.read_json(target_spec)
        if target_info['features_to_ignore'] is not None:
            cols_to_keep = [col for col in features.columns if col not in target_info['features_to_ignore']]
            features = features[cols_to_keep]

    # preprocess
    if preprocess:
        features = build_features.preprocess(
                        dataframe=features,   
                        cols_to_drop=feature_spec['cols_to_drop'],
                        clf_info=feature_spec['clf_info'],
                        cols_to_ignore=['Identifiers'],
                        threshold=False
                        )

    # combine features, targets, participants into one dataframe
    identifiers = features['Identifiers'].tolist()
    targets = pd.DataFrame(identifiers, columns=['Identifiers'])
    if target_spec is not None:
        targets = build_features.get_targets(
                                target_info=target_info, 
                                participants=identifiers
                                )   
    if drop_identifiers:
        features_final = features.merge(targets, on='Identifiers').drop(['Identifiers'], axis=1)
    else:
        features_final = features.merge(targets, on='Identifiers')
        
    # upsample minority class using smote 
    if oversample:
        features_final = build_features.smote(features_final)
    
    return features_final

