import warnings
warnings.filterwarnings("ignore")

def phenotype_features(
            feature_spec,
            participants=None,
            target_spec=None,
            drop_identifiers=True
            ):
    """make model features using whichever features are specified in "feature_spec" and whichever target specified in "target_spec"
    Features for selected "participants" are returned

    Args: 
        feature_spec (str or dict): full path to feature spec file OR dict loaded from file
        participants (list of str or None): (optional) list of participant identifiers (output from `make_dataset.get_participants`)
        target_spec (str or None): (optional) full path to target spec file. if None, then only features (X) are returned, else features (X) + target variable (y) are returned.
        drop_identifiers (bool): (optional) default is True (returns dataframe without 'Identifiers' column)

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

    # make `participants` is None is given
    if participants is None:
        participants_df = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'participants.csv'))
    else:
        participants_df = pd.DataFrame(participants, columns=['Identifiers'])

    # get assessment
    assessments = feature_spec['assessment']
    if feature_spec['assessment']=='all':
        assessments = ['Child Measures', 'Parent Measures', 'Teacher Measures']
    elif isinstance(feature_spec['assessment'], str):
        assessments = [feature_spec['assessment']]

    # get features (X)
    features = pd.DataFrame()
    for assess in assessments:
        feat = build_features.get_features(
                    assessment=assess,
                    domains=[feature_spec['domains']],
                    measures=[feature_spec['abbrevs']]
                    )
        features = pd.concat([features, feat])

    # filter based on participants
    features = features.merge(participants_df, on='Identifiers')
    
    # optionally add demographics as features (if there is a filename)
    add_features = feature_spec['add_features']
    if add_features['filename'] is not None:
        features_df = pd.read_csv(os.path.join(Defaults.FEATURE_DIR, add_features['filename']))
        tmp = features_df.merge(features, on=['Identifiers'])
        cols_to_factorize = add_features['cols_to_include']
        for col in cols_to_factorize:
            if col in tmp.columns and tmp[col].dtype=='object':
                tmp.loc[:, col] = tmp[col].factorize()[0]
        features = tmp

    # remove `features_to_ignore` from dataframe if any are provided in `target_spec`
    if target_spec is not None:
        target_info = io.read_json(target_spec)
        if target_info['features_to_ignore'] is not None:
            idx = features.columns.str.contains(('|'.join(target_info['features_to_ignore'])))
            features = features[features.columns[~idx]]

    # preprocess
    preprocessing = feature_spec['preprocessing']
    if preprocessing['preprocess']:
        features = build_features.preprocess(
                        dataframe=features,  
                        cols_to_drop=preprocessing['cols_to_drop'],
                        clf_info=preprocessing['clf_info'],
                        cols_to_ignore=preprocessing['cols_to_ignore'],
                        threshold=preprocessing['threshold']
                        )

    # combine features, targets, participants into one dataframe
    identifiers = features['Identifiers'].tolist()
    targets = pd.DataFrame(identifiers, columns=['Identifiers'])
    if target_spec is not None:
        targets = build_features.get_targets(
                                target_info=target_info, 
                                participants=identifiers
                                )   
    # drop identifiers from final feature matrix
    if drop_identifiers:
        features_final = features.merge(targets, on='Identifiers').drop(['Identifiers'], axis=1)
    else:
        features_final = features.merge(targets, on='Identifiers')

    # upsample minority class using smote 
    if target_spec is not None and preprocessing['preprocess'] and preprocessing['upsample']:
        if features_final.isnull().values.any(): # impute if there are NaN values
            features_final = build_features.column_transform(features_final, clf_info=preprocessing['clf_info'], cols_to_ignore=[target_info['outname']])
        x_cols = [col for col in features_final.columns if target_info['outname'] not in col]
        features_final = build_features.smote(y_train=features_final[[target_info['outname']]], X_train=features_final[x_cols])
    
    return features_final

