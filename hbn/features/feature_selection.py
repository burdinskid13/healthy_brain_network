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
        participants (list of str or None or pd dataframe): (optional) list of participant identifiers (output from `make_dataset.get_participants`) or pd dataframe containing column 'Identifiers' which contains participant identifiers and optionally containing 'participant_groups'
        target_spec (str or dict or None): (optional) full path to target spec file. if None, then only features (X) are returned, else features (X) + target variable (y) are returned.
        drop_identifiers (bool): (optional) default is True (returns dataframe without 'Identifiers' column)
    Returns: 
        features_final (pd dataframe): features to be input to modeling routine
    """
    import pandas as pd
    from hbn import io
    import os
    from hbn.constants import Defaults
    from hbn.features import build_features

    # load feature spec from json file
    if isinstance(feature_spec, str):
        feature_spec = io.read_json(feature_spec)

    # load target spec from json file
    if isinstance(target_spec, str):
        target_spec = io.read_json(target_spec)

    # make `participants` is None is given
    if participants is None:
        participants_df = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'participants.csv'))[['Identifiers']]
    elif isinstance(participants, pd.DataFrame):
        participants_df = participants[['Identifiers']]
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
        features = pd.concat([features, feat], axis=1)

    # merge features with participants
    features = build_features.drop_duplicates(dataframe=features)
    features = features.merge(participants_df, on='Identifiers')

    # optionally filter features (based on feature_spec)
    features = _filter_features(feature_spec, features)
    
    # optionally add features (based on feature_spec)
    features = _add_features(feature_spec, features)

    # optionally filter features (based on target_spec)
    if target_spec is not None:
        if target_spec['features_to_ignore'] is not None:
            idx = features.columns.str.contains(('|'.join(target_spec['features_to_ignore'])))
            features = features[features.columns[~idx]]

    # preprocess features
    preprocessing = feature_spec['preprocessing']
    if preprocessing['preprocess']:
        features = build_features.preprocess(
                        dataframe=features,  
                        cols_to_drop=preprocessing['cols_to_drop'],
                        clf_info=preprocessing['clf_info'],
                        cols_to_ignore=preprocessing['cols_to_ignore'],
                        threshold=preprocessing['threshold']
                        )

    # figure out if there are participant groups to be used in getting targets
    participant_groups = None
    if (isinstance(participants, pd.DataFrame)) and ('participant_groups' in participants.columns):
        participant_groups = participants['participant_groups'].tolist()

    # combine features, targets, participants into one dataframe
    identifiers = features['Identifiers'].tolist()
    targets = pd.DataFrame(identifiers, columns=['Identifiers'])
    if target_spec is not None:
        targets = build_features.get_targets(
                                target_info=target_spec, 
                                participants=identifiers,
                                participant_groups=participant_groups
                                )   
    # drop identifiers (and duplicates) from final feature matrix
    if drop_identifiers:
        features_final = features.merge(targets, on='Identifiers').drop(['Identifiers'], axis=1).drop_duplicates()
    else:
        features_final = features.merge(targets, on='Identifiers').drop_duplicates()

    # upsample minority class using smote 
    if target_spec is not None and preprocessing['preprocess'] and preprocessing['upsample']:
        if features_final.isnull().values.any(): # impute if there are NaN values
            features_final = build_features.column_transform(features_final, clf_info=preprocessing['clf_info'], cols_to_ignore=[target_spec['outname']])
        x_cols = [col for col in features_final.columns if target_spec['outname'] not in col]
        features_final = build_features.smote(y_train=features_final[[target_spec['outname']]], X_train=features_final[x_cols])
    
    return features_final


def _add_features(feature_spec, features):
    import pandas as pd
    import os
    from hbn.constants import Defaults

    # optionally add features (if there is a filename given)
    fname = feature_spec['add_features']['filename']
    cols_to_include = feature_spec['add_features']['columns']
    if fname is not None:
        features_df = pd.read_csv(os.path.join(Defaults.FEATURE_DIR, fname))
        tmp = features_df.merge(features, on=['Identifiers'])
        # factorize columns
        for col in cols_to_include:
            if col in tmp.columns and tmp[col].dtype=='object':
                tmp.loc[:, col] = tmp[col].factorize()[0]
        features = tmp
    
    return features


def _filter_features(feature_spec, features):
    import pandas as pd
    import os
    from hbn.constants import Defaults
    
    # optionally filter features (if there is a filename given)
    fname = feature_spec['filter_features']['filename']
    cols_to_filter = feature_spec['filter_features']['columns']
    if fname is not None:
        features_df = pd.read_csv(os.path.join(Defaults.SUBTYPE_DIR, fname))
        
        # loop over columns to filter
        features_filtered = pd.DataFrame()
        for filter_col in cols_to_filter:
            list_of_cols = features_df[features_df[filter_col]==True]['col_name'].dropna().unique()

            # loop over columns to filter
            for col in list_of_cols:
                if col in features.columns:
                    features_filtered.loc[:, col] = features[col]
        
        # merge 'Identifiers' back with filtered features
        features_filtered = pd.concat([features[['Identifiers']], features_filtered], axis=1)
        return features_filtered
    else:
        return features


def secondlevel_feature_selection(model_dir):
    """Load features from first-level modeling routine and extract top features, which are then used to create a new model spec
    (which can be used to run more models)
    Args: 
        model_dir (str): fullpath to model directory (e.g., "../<model_name>")
    Returns: 
        model_spec (list of str): fullpaths to new model spec files
    """
    import glob
    import pandas as pd
    from hbn import io
    from pathlib import Path
    import os
    from hbn.constants import Defaults

    # load spec info
    model_spec = glob.glob(os.path.join(model_dir, '*.json'))[0] # ASSUMES ONLY ONE MODEL PER DIRECTORY

    # load feature importances
    df_feat_importances = pd.read_csv(os.path.join(model_dir, 'classifier-feature_importance.csv'))

    # load model features
    features = glob.glob(os.path.join(model_dir, 'model_features_*'))[0]
    df_features = pd.read_csv(features)
    
    # loop over classifiers (if there are more than one)
    spec_info_all = []; spec_names = []
    classifiers = df_feat_importances['clf'].unique()
    for idx, clf in enumerate(classifiers):

        # get spec names
        spec_name = Path(model_spec).stem.split('-')[:-1]
        spec_name_new = '-'.join(spec_name) + '-' + clf + '-spec.json'
        spec_names.append(spec_name_new)

        # load spec info
        spec_info = io.read_json(model_spec)

        df1 = df_feat_importances[df_feat_importances['clf']==clf]
        # get top features from feature importances dataframe
        top_features = df1[df1['top_features']==True]['feature_names'].tolist()

        # get indices of top features
        indices = []
        for feat in top_features:
            indices.append(df_features.columns.get_loc(feat))

        # assign new features to spec file
        spec_info['x_indices'] = indices

        # update classifier
        spec_info['clf_info'] = [spec_info['clf_info'][idx]] ## FIX THIS LINE

        spec_info_all.append(spec_info)

    return spec_info_all, spec_names