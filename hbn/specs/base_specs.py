import os
import numpy as np

def pydraml():
    """Parameters in `base_info` and `spec_info` are required by pydraml pipeline: https://github.com/nipype/pydra-ml 
    """

    base_info = {
        "filename" : None,
        "x_indices" : None,
        "target_vars" : None,
        "permute" : [True, False],
        "group_var" : None,
        "n_splits" : 5,
        "test_size" : .2,
        "permute" : [True, False],
        "gen_feature_importance" : True,
        "gen_permutation_importance" : True,
        "permutation_importance_n_repeats" : 5,
        "permutation_importance_scoring" : "accuracy",
        "gen_shap" : False,
        "nsamples" : "auto",
        "l1_reg" : "aic",
        "plot_top_n_shap": 10,
        "metrics" : ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
        }

    spec_info = {
        'pydraml1':
        {'clf_info': 
        [
            ["sklearn.ensemble", "AdaBoostClassifier"],
            ["sklearn.naive_bayes", "GaussianNB"],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100}],
            ["sklearn.ensemble", "ExtraTreesClassifier", {"n_estimators": 100, "class_weight": "balanced"}],
            ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "liblinear", "penalty": "l1"}],
            ["sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}],
            ["sklearn.svm", "SVC", {"probability": True},
            [{"kernel": ["rbf", "linear"], "C": [1, 10, 100, 1000]}]],
        ]
        },
        'pydraml2':
        {'clf_info': 
        [
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]], # classifier has to be last list
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "saga", "penalty": "l1", "max_iter": 5000}]], # classifier has to be last list
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50}]] # classifier has to be last list
        ],
        },
        'pydraml3':
        {'clf_info': 
        [
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 10}]], # classifier has to be last list
        ],
        },
        }
    return base_info, spec_info


def participants():
    """Any variables set in `spec_info` (e.g., `age`, `sex`, `race`) should be present in `base_info.filename` 
    """

    def _concat_dicts(*dicts):
        new_dict = {}
        for d in dicts:
            new_dict.update(d)
        
        return new_dict

    base_info = {'filename': 'participant_train_test.csv',
                'participant_id': 'Identifiers', # should be string (e.g., 'Identifiers', 'pat_id') and it's assumed that the `participant_id` column is in the features and targets dataframes (code checks this)
                'split': ['train', 'test'],
                'train_test_split': {'split_col': 'split', 'train': 'train', 'test': 'test'}
                }

    # spec files
    spec_info_adhd = {
        'participants-adhd-all':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given']
        },
        'participants-adhd-male':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given']
        },
        'participants-adhd-female':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['female'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given']
        },
        'participants-adhd-black':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-white':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Causasian'],
        },
        'participants-adhd-black-male':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-white-male':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Causasian'],
        },
        'participants-adhd-black-female':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['female'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-white-female':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['female'],
         'DX_Cat_Name': ['ADHD', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Causasian'],
        },
        }
    spec_info_asd = {
        'participants-asd-all':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given']
        },
        'participants-asd-male':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given']
        },
        'participants-asd-female':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['female'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given']
        },
        'participants-asd-black':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-asd-white':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Causasian'],
        },
        'participants-asd-black-male':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-asd-white-male':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['male'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Causasian'],
        },
        'participants-asd-black-female':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['female'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-asd-white-female':
        {
         'Age_round': [int(t) for t in np.arange(5,22)],
         'Sex': ['female'],
         'DX_Cat_Name': ['Autism Spectrum Disorder', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Causasian'],
        },
        }
    spec_info_reading = {
        'participants-Reading-all':
        {
         'Age_round': [int(t) for t in np.arange(6,22)],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-male':
        {
         'Age_round': [int(t) for t in np.arange(6,22)],
         'Sex': ['male'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-female':
        {
         'Age_round': [int(t) for t in np.arange(6,22)],
         'Sex': ['female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-early-readers':
        {
         'Age_round': [6,7,8],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-emerging-readers':
        {
         'Age_round': [9,10],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-fluent-readers':
        {
         'Age_round': [11,12,13,14,15,16,17,18],
         'Sex': ['male', 'female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-early-readers-female':
        {
         'Age_round': [6,7,8],
         'Sex': ['female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-emerging-readers-female':
        {
         'Age_round': [9,10],
         'Sex': ['female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-fluent-readers-female':
        {
         'Age_round': [11,12,13,14,15,16,17,18],
         'Sex': ['female'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-early-readers-male':
        {
         'Age_round': [6,7,8],
         'Sex': ['male'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-emerging-readers-male':
        {
         'Age_round': [9,10],
         'Sex': ['male'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-fluent-readers-male':
        {
         'Age_round': [11,12,13,14,15,16,17,18],
         'Sex': ['male'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-Black':
        {
         'Age_round': [int(t) for t in np.arange(6,22)],
         'Sex': ['male', 'female'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-White':
        {
         'Age_round': [int(t) for t in np.arange(6,22)],
         'Sex': ['male', 'female'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Causasian'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-multiple-races':
        {
         'Age_round': [int(t) for t in np.arange(6,22)],
         'Sex': ['male', 'female'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Two or more races'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        'participants-Reading-Hispanic':
        {
         'Age_round': [int(t) for t in np.arange(6,22)],
         'Sex': ['male', 'female'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Hispanic'],
         'DX_Cat_Name': ['Specific Learning Disorder with Impairment in Reading', 'No Diagnosis Given']
        },
        }

    # concat dicts
    spec_info = _concat_dicts(spec_info_adhd, spec_info_asd, spec_info_reading)


    return base_info, spec_info


def targets():
    """hardcode target features
    """

    base_info = {
        'upsample': True, # upsample minority target class using SMOTE
                    }
    spec_info = {'target-Diagnosis':
                    {
                    'filename': 'all_participant_diagnoses.csv',
                    'target_column': 'DX_Cat_Name', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'DX_Cat_Name'], # columns we want in the final dataframe
                    },
                }

    return base_info, spec_info


def features():

    def _concat_dicts(*dicts):
        new_dict = {}
        for d in dicts:
            new_dict.update(d)
        
        return new_dict
    
    base_info = {
                "threshold": False, # threshold dataframe based on some fixed criterion. We are using 50% for columns and 20% for rows. If threshold is False, then only NaN entries are removed (no thresholding applied)
                "clf_info": {
                    "numeric": [
                        [
                            "sklearn.impute",
                            "SimpleImputer",
                            {
                                # "strategy": "constant",
                                # "fill_value": None
                                "strategy": "mean",
                                "add_indicator": True
                            }
                        ],
                        [
                            "sklearn.preprocessing",
                            "StandardScaler",
                            {}
                        ]
                    ],
                    "category": [
                        [
                            "sklearn.impute",
                            "SimpleImputer",
                            {
                                "strategy": "constant", # most_frequent,
                                "fill_value": None,
                                "add_indicator": True
                            }
                        ],
                        [
                            "sklearn.preprocessing",
                            "OneHotEncoder",
                            {
                                "handle_unknown": "ignore",
                                "sparse": False, # sparse_output
                                "categories": 'auto',
                                "drop": 'if_binary',
                                "max_categories": 5
                            }
                        ]
                    ]
                    }
        }

    all_info = {
            'features-Child':
            {
            "filename": 'Child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": None
            }, 
            'features-Parent':
            {
            "filename": 'Parent-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": None
            },
            'features-Teacher':
            {
            "filename": 'Teacher-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": None
            },
            'features-Child-remove-total-scores':
            {
            "filename": 'Child-features-Not_Total_Scores-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID',  'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": None
            }, 
            'features-Parent-remove-total-scores':
            {
            "filename": 'Parent-features-Not_Total_Scores-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season',  'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": None
            },
            'features-Teacher-remove-total-scores':
            {
            "filename": 'Teacher-features-Not_Total_Scores-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season',  'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": None
            },
            'features-all-demos':
            {
            "filename": 'all-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season',  'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'Sex', 'Age_round', 'PreInt_Demos_Fam,Child_Race_cat', 'PreInt_Demos_Fam,Child_Ethnicity_cat']
            }
            }
        
    reading_info = {
                'features-Child-language':
                {
                "filename": 'Child-features-Not_Total_Scores-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CELF', 'PPVT', 'EVT']
                }, 
                'features-Child-phonological':
                {
                "filename": 'Child-features-Not_Total_Scores-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CTOPP']
                },
                'features-Child-production':
                {
                "filename": 'Child-features-Not_Total_Scores-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'GFTA']
                },
                'features-Child-executive-function':
                {
                "filename": 'Child-features-Not_Total_Scores-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'NIH']
                }, 
                'features-Child-intelligence':
                {
                "filename": 'Child-features-Not_Total_Scores-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'WISC', 'WAIS', 'KBIT']
                }, 
                'features-Child-achievement':
                {
                "filename": 'Child-features-Not_Total_Scores-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'WIAT', 'TOWRE']
                }, 
                'features-Child-emotional-status':
                {
                "filename": 'Child-features-Not_Total_Scores-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR', 'C3SR', 'SCARED_SR', 'CIS_SR', 'WHODAS_SR', 'PANAS']
                }
                }

    internalizing_externalizing = {
                'features-Child-internalizing':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_Int']
                }, 
                'features-Child-externalizing':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_Ext']
                }, 
                'features-Parent-internalizing':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_Int', 'Internalising']
                }, 
                'features-Parent-externalizing':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_Ext', 'Externalising']
                }, 
                'features-Teacher-internalizing':
                {
                "filename": 'Teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_Int']
                }, 
                'features-Teacher-externalizing':
                {
                "filename": 'Teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_Ext']
                },
                'features-all-externalizing':
                {
                "filename": 'all-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_Ext', 'CBCL_Ext', 'Externalising', 'YSR_Ext']
                },
                'features-all-internalizing':
                {
                "filename": 'all-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_Int', 'CBCL_Int', 'Internalising', 'YSR_Int']
                } 
                }

    adhd_info = {                
                'features-child-connors':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'C3SR']
                }, 
                'features-child-cbcl':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR', 'ASR']
                }, 
                'features-child-anxiety':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SCARED_SR']
                }, 
                'features-child-mood':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'MFQ_SR', 'PANAS']
                }, 
                'features-child-suicide':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CSSRS']
                }, 
                'features-child-language':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CELF_Full_5to8', 'CELF_Full_9to21', 'CELF_Meta', 'EVT', 'PPVT', 'GFTA', 'CTOPP', 'TOWRE', 'CELF']
                }, 
                'features-child-intelligence':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'WISC', 'WAIS', 'WAIS_Abb', 'KBIT']
                }, 
                'features-child-achievement':
                {
                "filename": 'Child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'WIAT']
                }, 
                'features-parent-cbcl':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL', 'CBCL_Pre']
                }, 
                'features-parent-anxiety':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SCARED_P']
                }, 
                'features-parent-strengths-weaknesses-adhd':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SWAN', 'ESWAN', 'SDQ']
                }, 
                }
    
    asd_info = {
                'features-parent-asd':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'ASSQ']
                }, 
                'features-parent-social-communication':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SCQ', 'SAS', 'SRS', 'SRS_Pre']
                }, 
                'features-parent-child-mind-institute':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SympChck']
                }, 
                'features-parent-mood':
                {
                "filename": 'Parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'MFQ_P']
                }
    }

    spec_info = _concat_dicts(all_info, reading_info, adhd_info, asd_info, internalizing_externalizing)
    
    return base_info, spec_info




