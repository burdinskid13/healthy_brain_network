# useful functions for manipulating data

import pandas as pd
import logging
from hbn.constants import Defaults
from pathlib import Path
import os
import glob

def remove_small_groups(dataframe, columns_to_stratify, min_group_size=2):
    """Removes the rows from the dataframe where the group size is less than min_group_size

    Args:
        dataframe (pd.DataFrame): The dataframe to remove the rows from
        columns_to_stratify (list): A list of columns to use for grouping

    Returns:
        pd.DataFrame: The dataframe with the small groups removed
    """

    # Group the dataframe by the columns_to_stratify columns
    grouped_dataframe = dataframe.groupby(columns_to_stratify)

    # Filter the grouped dataframe to only include groups with at least min_group_size rows
    filtered_dataframe = grouped_dataframe.filter(lambda x: len(x) >= min_group_size)

    # Remove the rows from the dataframe that are not in the filtered dataframe
    dataframe = dataframe[dataframe.index.isin(filtered_dataframe.index)]

    # Return the filtered dataframe
    return dataframe.reset_index(drop=True)

def get_summary_df(fname):

    df_summary = pd.DataFrame()
    if os.path.isfile(fname):
        df_summary = pd.read_csv(fname, engine='python')

        # make new columns for visualization
        df_summary['data'] = df_summary['data'].str.replace('model-', '')
        if 'Age_round' in df_summary.columns:
            df_summary['readers'] = df_summary['Age_round'].map({'6_7_8': 'early', 
                                                                '9_10': 'emerging', 
                                                                '11_12_13_14_15_16_17_18': 'expert',
                                                                '6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21': 'all'
                                                                })
            df_summary['development'] = df_summary['Age_round'].map({'5_6_7': '5-7',
                                                                    '8_9_10': '8-10',
                                                                    '11_12_13': '11-13',
                                                                    '14_15_16': '14-16',
                                                                    '17_18_19_20_21': '17-21',
                                                                    '5_6_7_8_9_10_11_12_13_14_15_16_17': 'all',
                                                                    '5_6_7_8_9': '5-9',
                                                                    '10_11_12_13': '10-13',
                                                                    '14_15_16_17_18': '14-18',
                                                                    '5_6_7_8_9_10_11': '5-11',
                                                                    '12_13_14_15_16_17': '12-17',
                                                                    '5_6_7_8_9_10_11_12_13_14_15_16_17': 'all'
                                                                    })
        elif 'puberty' in df_summary.columns:
            df_summary['development'] = df_summary['puberty'].map({'pre': 'pre-puberty', 'post': 'post-puberty'})
            
        df_summary['features'] = df_summary['feat_spec_name'].str.replace('features-', '')
        df_summary = df_summary.rename(columns={'PreInt_Demos_Fam,Child_Race_cat': 'Race'})

        # if Race is NaN, assign 'all'
        if 'Race' not in df_summary.columns:
            df_summary['Race'] ='all'

    return df_summary

def get_feature_df(fname):
    
    df_feat = pd.DataFrame()
    if os.path.isfile(fname):
        df_feat = pd.read_csv(fname, engine='python')

        # make new columns for visualization
        if 'Age_round' in df_feat.columns:
            df_feat['readers'] = df_feat['Age_round'].map({'6_7_8': 'early', 
                                                            '9_10': 'emerging', 
                                                            '11_12_13_14_15_16_17_18': 'expert',
                                                            '6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21': 'all'
                                                            })
            df_feat['development'] = df_feat['Age_round'].map({'5_6_7': '5-7',
                                                            '8_9_10': '8-10',
                                                            '11_12_13': '11-13',
                                                            '14_15_16': '14-16',
                                                            '17_18_19_20_21': '17-21',
                                                            '5_6_7_8_9': '5-9',
                                                            '10_11_12_13': '10-13',
                                                            '14_15_16_17_18': '14-18',
                                                            '5_6_7_8_9_10_11': '5-11',
                                                            '12_13_14_15_16_17': '12-17',
                                                            '5_6_7_8_9_10_11_12_13_14_15_16_17': 'all'
                                                            })
        elif 'puberty' in df_feat.columns:
            df_feat['development'] = df_feat['puberty'].map({'pre': 'pre-puberty', 'post': 'post-puberty'})
            
        df_feat['features'] = df_feat['feat_spec_name'].str.replace('features-', '')
        df_feat['feature_names'] = df_feat['feature_importances_names'].str.split(',').str.get(1)
        df_feat = df_feat.rename(columns={'PreInt_Demos_Fam,Child_Race_cat': 'Race'})

        # if Race is NaN, assign 'all'
        if 'Race' not in df_feat.columns:
            df_feat['Race'] ='all'

    return df_feat

def load_model_results(model='reading_july'):
    """load all model results for `model`: summary and feature importances, and optionally save out
    """

    # get all models run in `model`
    models = glob.glob(os.path.join(Defaults.MODEL_DIR, model, '*', '*'))

    summary_all = pd.DataFrame()
    feat_all = pd.DataFrame()
    for model in models:

        # read model summary and features into pd dataframe
        df_summary = get_summary_df(fname=os.path.join(model, 'model-summary.csv'))
        df_feature = get_feature_df(os.path.join(model, 'feature_importance.csv'))

        # append dataframe
        summary_all = pd.concat([summary_all, df_summary])
        feat_all = pd.concat([feat_all, df_feature])
    
    return summary_all.reset_index(drop=True), feat_all.reset_index(drop=True)


def calculate_topic_representation(model):

    keys = None; df_concat = pd.DataFrame()
    df_features_selected = get_feature_df(os.path.join(model, 'feature_importance.csv'))
    if 'feature_importances_names' in df_features_selected.columns:
        keys = pd.DataFrame(df_features_selected['feature_importances_names'] \
                            .str.replace('numeric__', '') \
                                .str.replace('category__', '')\
                                    .str.split(',').str.get(0)) \
                                        .rename({'feature_importances_names': 'count'}, axis=1)

    # get keys across selected features
    if keys is not None:
        df_keys = keys['count'].value_counts().reset_index()
        df_keys['perc'] = round(df_keys['count'] / len(keys) * 100, 2)
        df_concat = pd.concat([df_features_selected.head(len(df_keys)), df_keys], axis=1)
        df_concat_selected = df_concat.drop(columns=['feature_importances', 'feature_importances_names', 'feature_names'])

        # get keys across overall features
        df_features_overall = pd.read_csv(os.path.join(model, 'features-train.csv'))
        keys = pd.DataFrame(df_features_overall.columns \
                    .str.replace('numeric__', '') \
                        .str.replace('category__', '')[1:] \
                            .str.split(',') \
                                .str.get(0), columns=['count'])

        df_keys = keys['count'].value_counts().reset_index()
        df_keys['perc'] = round(df_keys['count'] / len(keys) * 100, 2)
        df_base = df_features_selected.head(1)
        df_base['merge'] = 'common'
        df_keys['merge'] = 'common'
        df_concat_overall = df_keys.merge(df_base, on='merge') \
            .drop(columns=['feature_importances', 'feature_importances_names', 'feature_names', 'merge'])
        df_concat_overall['feature_threshold'] = 'overall' + '-' + df_features_selected['feature_threshold'].unique()[0]

        df_concat = pd.concat([df_concat_overall, df_concat_selected])
        df_concat.rename(columns={'index': 'key'}, inplace=True)

    return df_concat


def load_topic_representation(model='reading_july'):
    """"load model features input to `model`"""
    from hbn.visualization import utils as utils

    # get all models run in `model`
    models = glob.glob(os.path.join(Defaults.MODEL_DIR, model, '*', '*'))

    # get % of topics across all features input to model
    df_keys_all = pd.DataFrame()
    for model in models:
        df_keys = calculate_topic_representation(model=model)
        df_keys_all = pd.concat([df_keys_all, df_keys])

    # remap dx
    dx_cols = [col for col in df_keys_all.columns if 'DX' in col]
    for dx in dx_cols:
        df_keys_all[dx] = df_keys_all[dx].map(utils.remap_dx())

    return df_keys_all


def load_model_features(model='reading_july'): 
    import os
    from hbn.visualization import utils as utils
    
    # load data for all models run in `model_parent`
    _, features = load_model_results(model=model)

    # remap diagnosis
    dx_cols = [col for col in features.columns if 'DX' in col]
    for dx in dx_cols:
        features[dx] = features[dx].map(utils.remap_dx())

    # get keys
    df_items = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'item-names-cleaned.csv'))
    features = features.merge(df_items[['keys', 'datadic']], left_on='feature_names', right_on='keys')

    return features