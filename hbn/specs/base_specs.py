import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def pydraml():
    """Parameters in `base_info` and `spec_info` are required by pydraml pipeline: https://github.com/nipype/pydra-ml 
    """

    base_info = {
        "filename" : None,
        "x_indices" : None,
        "target_vars" : None,
        "permute" : [True, False],
        "group_var" : None,
        "n_splits" : 10,
        "test_size" : .2,
        "permute" : [True, False],
        "oversample": True,
        "feature_selection": True,
        "feature_selection_strategy": 'intersection', # 'intersection or 'union'
        "gen_feature_importance" : True,
        "gen_permutation_importance" : True,
        "permutation_importance_n_repeats" : 5,
        "permutation_importance_scoring" : "accuracy",
        "gen_shap" : False,
        "nsamples" : "auto",
        "l1_reg" : "aic",
        "plot_top_n_shap": 10,
        "metrics" : ['roc_auc_score'] 
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
            ["sklearn.ensemble", "ExtraTreesClassifier", {"n_estimators": 10, "class_weight": "balanced"}]], # classifier has to be last list
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "saga", "penalty": "l1", "max_iter": 100}]], # classifier has to be last list
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50}]], # classifier has to be last list
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.svm", "LinearSVC"]], # classifier has to be last list
        ],
        },
        'pydraml3':
        {'clf_info': 
        # [
        # [["sklearn.impute", "SimpleImputer", {"strategy": "mean", "add_indicator": True},
        #    "sklearn.preprocessing", "StandardScaler"],
        #    ["sklearn.impute", "SimpleImputer", {"strategy": "constant", "fill_value": None, "add_indicator": True},
        #     "sklearn.preprocessing", "OneHotEncoder", {"handle_unknown": "ignore", "sparse_output": False, "categories": 'auto', "drop": 'if_binary', "min_frequency": 0.01},
        #         "sklearn.preprocessing", "StandardScaler"],
        #     ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50}]], # classifier has to be last list
        # ],
        [
            [["sklearn.impute", "SimpleImputer"],
            ["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50}]],
        ],
        },
        'pydraml4':
        {'clf_info': 
        [
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.linear_model","RidgeCV",{"fit_intercept": True,"normalize": True}]], # classifier has to be last list
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
                }
    
    age = 'age_round'
    sex = 'sex'
    race = 'PreInt_Demos_Fam,Child_Race_cat'
    dx_reading = 'DX_Reading'
    dx_adhd = 'DX_ADHD'
    
    # spec files
    spec_info_adhd_all_comorbidities = {
        'participants-adhd-all':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-male-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-male-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-female-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-female-postpubertal':
        {
         'puberty': ['post'],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage1':
        {
         age: [5,6,7],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage2':
        {
         age: [8,9,10],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage3':
        {
         age: [11,12,13],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage4':
        {
         age: [14,15,16],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage5':
        {
         age: [17,18,19,20,21],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage1-male':
        {
         age: [5,6,7],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage2-male':
        {
         age: [8,9,10],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage3-male':
        {
         age: [11,12,13],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage4-male':
        {
         age: [14,15,16],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage5-male':
        {
         age: [17,18,19,20,21],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage1-female':
        {
         age: [5,6,7],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage2-female':
        {
         age: [8,9,10],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage3-female':
        {
         age: [11,12,13],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage4-female':
        {
         age: [14,15,16],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-stage5-female':
        {
         age: [17,18,19,20,21],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-all-black':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-all-white':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Caucasian'],
        },
        'participants-adhd-all-black-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-all-white-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Caucasian'],
        },
        'participants-adhd-all-black-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-all-white-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_all_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Caucasian'],
        },
        }
    spec_info_adhd_no_comorbidities = {
        'participants-adhd-only':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-male-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['male'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-female-prepubertal':
        {
        'puberty': ['pre'],
         sex: ['female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-male-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-female-postpubertal':
        {
        'puberty': ['post'],
         sex: ['female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-stage1':
        {
         age: [5,6,7],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-stage2':
        {
         age: [8,9,10],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-stage3':
        {
         age: [11,12,13],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-stage4':
        {
         age: [14,15,16],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-stage5':
        {
         age: [17,18,19,20,21],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-adhd-only-black':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-only-white':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Caucasian'],
        },
        'participants-adhd-only-black-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-only-white-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Caucasian'],
        },
        'participants-adhd-only-black-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['Black/African American'],
        },
        'participants-adhd-only-white-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         dx_adhd: ['adhd_no_comorbidities', 'No Diagnosis Given'],
         'PreInt_Demos_Fam,Child_Race_cat': ['White/Caucasian'],
        },
        }
    spec_info_adhd_other = {
        'participants-adhd-combined_type':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['male', 'female'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['male', 'female'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male', 'female'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male', 'female'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-male-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['male'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-male-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-male-prepubertal':
        {
        'puberty': ['pre'],
         sex: ['male'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-male-postpubertal':
        {
         'puberty': ['post'],
         sex: ['male'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-female-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['female'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-combined_type-female-postpubertal':
        {
         'puberty': ['post'],
         sex: ['female'],
         'DX_Subtype_Name': ['ADHD-Combined Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-female-prepubertal':
        {
         'puberty': ['pre'],
         sex: ['female'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        'participants-adhd-inattentive_type-female-postpubertal':
        {
         'puberty': ['post'],
         sex: ['female'],
         'DX_Subtype_Name': ['ADHD-Inattentive Type', 'No Diagnosis Given']
        },
        }

    spec_info_anxiety_adhd = {
        'participants-adhd-combined_type-anxiety-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         'DX_Anxiety': ['ADHD-Combined Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-inattentive_type-anxiety-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         'DX_Anxiety': ['ADHD-Inattentive Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-combined_type-anxiety-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         'DX_Anxiety': ['ADHD-Combined Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-inattentive_type-anxiety-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         'DX_Anxiety': ['ADHD-Inattentive Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-inattentive_type-anxiety-male-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['male'],
         'DX_Anxiety': ['ADHD-Inattentive Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-inattentive_type-anxiety-male-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['male'],
         'DX_Anxiety': ['ADHD-Inattentive Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-combined_type-anxiety-male-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['male'],
         'DX_Anxiety': ['ADHD-Combined Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-combined_type-anxiety-male-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['male'],
         'DX_Anxiety': ['ADHD-Combined Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-inattentive_type-anxiety-female-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['female'],
         'DX_Anxiety': ['ADHD-Inattentive Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-inattentive_type-anxiety-female-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['female'],
         'DX_Anxiety': ['ADHD-Inattentive Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-combined_type-anxiety-female-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['female'],
         'DX_Anxiety': ['ADHD-Combined Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        'participants-adhd-combined_type-anxiety-female-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['female'],
         'DX_Anxiety': ['ADHD-Combined Type (no Anxiety)', 'Anxiety (no ADHD)']
        },
        }
    spec_info_depression_adhd = {
        'participants-adhd-combined_type-depression-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         'DX_Depression': ['ADHD-Combined Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-inattentive_type-depression-male':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male'],
         'DX_Depression': ['ADHD-Inattentive Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-combined_type-depression-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         'DX_Depression': ['ADHD-Combined Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-inattentive_type-depression-female':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['female'],
         'DX_Depression': ['ADHD-Inattentive Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-inattentive_type-depression-male-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['male'],
         'DX_Depression': ['ADHD-Inattentive Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-inattentive_type-depression-male-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['male'],
         'DX_Depression': ['ADHD-Inattentive Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-combined_type-depression-male-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['male'],
         'DX_Depression': ['ADHD-Combined Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-combined_type-depression-male-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['male'],
         'DX_Depression': ['ADHD-Combined Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-inattentive_type-depression-female-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['female'],
         'DX_Depression': ['ADHD-Inattentive Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-inattentive_type-depression-female-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['female'],
         'DX_Depression': ['ADHD-Inattentive Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-combined_type-depression-female-young':
        {
         age: [5,6,7,8,9,10,11],
         sex: ['female'],
         'DX_Depression': ['ADHD-Combined Type (no Depression)', 'Depression (no ADHD)']
        },
        'participants-adhd-combined_type-depression-female-old':
        {
         age: [12,13,14,15,16,17],
         sex: ['female'],
         'DX_Depression': ['ADHD-Combined Type (no Depression)', 'Depression (no ADHD)']
        },
        }

    spec_info_gender_adhd_all_comorbidities = {
        'participants-adhd-all-comorbidities':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities']
        },
        'participants-adhd-all-comorbidities-black':
        {
        age: [int(t) for t in np.arange(5,18)],
        sex: ['male', 'female'],
        dx_adhd: ['adhd_all_comorbidities'],
        race: ['Black/African American'],
        },
        'participants-adhd-all-comorbidities-white':
        {
        age: [int(t) for t in np.arange(5,18)],
        sex: ['male', 'female'],
        dx_adhd: ['adhd_all_comorbidities'],
        race: ['White/Caucasian'],
        },
        'participants-adhd-all-comorbidities-stage1':
        {
         age: [5,6,7],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities']
        },
        'participants-adhd-all-comorbidities-stage2':
        {
         age: [8,9,10],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities']
        },
        'participants-adhd-all-comorbidities-stage3':
        {
         age: [11,12,13],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities']
        },
        'participants-adhd-all-comorbidities-stage4':
        {
         age: [14,15,16],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities']
        },
        'participants-adhd-all-comorbidities-stage5':
        {
         age: [17,18,19,20,21],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities'],
        },
        'participants-adhd-all-comorbidities-prepuberty-black':
        {
        age: [5,6,7,8,9],
        sex: ['male', 'female'],
        dx_adhd: ['adhd_all_comorbidities'],
        race: ['Black/African American'],
        },
        'participants-adhd-all-comorbidities-puberty-black':
        {
        age: [10,11,12,13],
        sex: ['male', 'female'],
        dx_adhd: ['adhd_all_comorbidities'],
        race: ['Black/African American'],
        },
        'participants-adhd-all-comorbidities-postpuberty-black':
        {
        age: [14,15,16,17,18],
        sex: ['male', 'female'],
        dx_adhd: ['adhd_all_comorbidities'],
        race: ['Black/African American'],
        },
        'participants-adhd-all-comorbidities-prepuberty-white':
        {
         age: [5,6,7,8,9],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities'],
         race: ['White/Caucasian'],
        },
        'participants-adhd-all-comorbidities-puberty-white':
        {
         age: [10,11,12,13],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities'],
         race: ['White/Caucasian'],
        },
        'participants-adhd-all-comorbidities-postpuberty-white':
        {
         age: [14,15,16,17,18],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_all_comorbidities'],
         race: ['White/Caucasian'],
        },
    }
    spec_info_gender_adhd_no_comorbidities = {
        'participants-adhd-no-comorbidities':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities']
        },
        'participants-adhd-no-comorbidities-black':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities'],
         race: ['Black/African American'],
        },
        'participants-adhd-no-comorbidities-white':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities'],
         race: ['White/Caucasian'],
        },
        'participants-adhd-no-comorbidities-stage1':
        {
         age: [5,6,7],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities']
        },
        'participants-adhd-no-comorbidities-stage2':
        {
         age: [8,9,10],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities']
        },
        'participants-adhd-no-comorbidities-stage3':
        {
         age: [11,12,13],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities']
        },
        'participants-adhd-no-comorbidities-stage4':
        {
         age: [14,15,16],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities']
        },
        'participants-adhd-no-comorbidities-stage5':
        {
         age: [17,18,19,20,21],
         sex: ['male', 'female'],
         dx_adhd: ['adhd_no_comorbidities']
        },
    }
    spec_info_gender_other = {
        'participants-other_diagnoses':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['other_diagnoses']
        },
        'participants-no_diagnosis_given':
        {
         age: [int(t) for t in np.arange(5,18)],
         sex: ['male', 'female'],
         dx_adhd: ['No Diagnosis Given']
        },
        'participants-adhd_combined':
        {
        age: [int(t) for t in np.arange(5,18)],
        sex: ['male', 'female'],
        'DX_Subtype_Name': ['ADHD-Combined Type']
        },
        'participants-adhd_inattentive':
        {
        age: [int(t) for t in np.arange(5,18)],
        sex: ['male', 'female'],
        'DX_Subtype_Name': ['ADHD-Inattentive Type']
        },
        'participants-adhd_hyperactive':
        {
        age: [int(t) for t in np.arange(5,18)],
        sex: ['male', 'female'],
        'DX_Subtype_Name': ['ADHD-Hyperactive/Impulsive Type']
        },
    }
    
    spec_info_gender_reading_all_comorbidities = {
        'participants-reading-all-comorbidities':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities']
        },
        'participants-No-Diagnosis':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['No Diagnosis Given']
        },
        'participants-reading-all-comorbidities-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities']
        },
        'participants-reading-all-comorbidities-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities']
        },
        'participants-reading-all-comorbidities-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities']
        },
        'participants-reading-all-comorbidities-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities'],
          race: ['Black/African American'],
        },
        'participants-reading-all-comorbidities-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities'],
          race: ['White/Caucasian'],
        },
        }
    spec_info_gender_reading_no_comorbidities = {
        'participants-reading-no-comorbidities':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities']
        },
        'participants-No-Diagnosis':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['No Diagnosis Given']
        },
        'participants-reading-no-comorbidities-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities']
        },
        'participants-reading-no-comorbidities-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities']
        },
        'participants-reading-no-comorbidities-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities']
        },
        'participants-reading-no-comorbidities-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities'],
          race: ['Black/African American'],
        },
        'participants-reading-no-comorbidities-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities'],
          race: ['White/Caucasian'],
        },
        }
    spec_info_gender_reading_adhd = {
        'participants-reading-adhd-only':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd']
        },
        'participants-reading-adhd-only-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd']
        },
        'participants-reading-adhd-only-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd']
        },
        'participants-reading-adhd-only-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd']
        }
        }

    spec_info_reading_all_comorbidites = {
        'participants-reading-all':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-diagnoses':
        {
         age: [int(t) for t in np.arange(6,18)],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities', 'reading_no_comorbidities', 
                      'reading_adhd',  'adhd_no_reading', 
                      'No Diagnosis Given', 'no_reading', 'reading_anxiety']
        },
        'participants-reading-all-male':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-female':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-early-readers-female':
        {
         age: [6,7,8],
         sex: ['female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-emerging-readers-female':
        {
         age: [9,10],
         sex: ['female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-fluent-readers-female':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['female'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-early-readers-male':
        {
         age: [6,7,8],
         sex: ['male'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-emerging-readers-male':
        {
         age: [9,10],
         sex: ['male'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-fluent-readers-male':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-multiple-races':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Two or more races'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-Hispanic':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Hispanic'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-early-readers-Black':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-early-readers-White':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-emerging-readers-Black':
        {
         age: [9,10],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-emerging-readers-White':
        {
         age: [9,10],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-fluent-readers-Black':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-all-fluent-readers-White':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_all_comorbidities', 'No Diagnosis Given']
        },
        }
    spec_info_reading_no_comorbidities = {
        'participants-reading-only':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-male':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-female':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-early-readers-female':
        {
         age: [6,7,8],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-emerging-readers-female':
        {
         age: [9,10],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-fluent-readers-female':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-early-readers-male':
        {
         age: [6,7,8],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-emerging-readers-male':
        {
         age: [9,10],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-fluent-readers-male':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-multiple-races':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Two or more races'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        'participants-reading-only-Hispanic':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Hispanic'],
         dx_reading: ['reading_no_comorbidities', 'No Diagnosis Given']
        },
        }  
    spec_info_reading_adhd = {
        'participants-reading-adhd':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-male':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-female':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-early-readers-female':
        {
         age: [6,7,8],
         sex: ['female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-emerging-readers-female':
        {
         age: [9,10],
         sex: ['female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-fluent-readers-female':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['female'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-early-readers-male':
        {
         age: [6,7,8],
         sex: ['male'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-emerging-readers-male':
        {
         age: [9,10],
         sex: ['male'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-fluent-readers-male':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-multiple-races':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Two or more races'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        'participants-reading-adhd-Hispanic':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Hispanic'],
         dx_reading: ['reading_adhd', 'No Diagnosis Given']
        },
        }
    spec_info_reading_anxiety = {
        'participants-reading-anxiety':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-male':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-female':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-early-readers-female':
        {
         age: [6,7,8],
         sex: ['female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-emerging-readers-female':
        {
         age: [9,10],
         sex: ['female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-fluent-readers-female':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['female'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-early-readers-male':
        {
         age: [6,7,8],
         sex: ['male'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-emerging-readers-male':
        {
         age: [9,10],
         sex: ['male'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-fluent-readers-male':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-multiple-races':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Two or more races'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        'participants-reading-anxiety-Hispanic':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Hispanic'],
         dx_reading: ['reading_anxiety', 'No Diagnosis Given']
        },
        }
    spec_info_adhd_no_reading = {
        'participants-no-reading':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-male':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-female':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-early-readers-female':
        {
         age: [6,7,8],
         sex: ['female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-emerging-readers-female':
        {
         age: [9,10],
         sex: ['female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-fluent-readers-female':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['female'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-early-readers-male':
        {
         age: [6,7,8],
         sex: ['male'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-emerging-readers-male':
        {
         age: [9,10],
         sex: ['male'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-fluent-readers-male':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-multiple-races':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Two or more races'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        'participants-no-reading-Hispanic':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Hispanic'],
         dx_reading: ['adhd_no_reading', 'No Diagnosis Given']
        },
        }
    spec_info_reading_anxiety2 = {
        'participants-reading_anxiety2':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-male':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-female':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-early-readers-female':
        {
         age: [6,7,8],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-emerging-readers-female':
        {
         age: [9,10],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-fluent-readers-female':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-early-readers-male':
        {
         age: [6,7,8],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-emerging-readers-male':
        {
         age: [9,10],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-fluent-readers-male':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-multiple-races':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Two or more races'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        'participants-reading_anxiety2-Hispanic':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Hispanic'],
         dx_reading: ['reading_no_comorbidities', 'reading_anxiety']
        },
        }
    spec_info_reading_adhd2 = {
        'participants-reading_adhd2':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-male':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-female':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-early-readers':
        {
         age: [6,7,8],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-emerging-readers':
        {
         age: [9,10],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-fluent-readers':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male', 'female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-early-readers-female':
        {
         age: [6,7,8],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-emerging-readers-female':
        {
         age: [9,10],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-fluent-readers-female':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['female'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-early-readers-male':
        {
         age: [6,7,8],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-emerging-readers-male':
        {
         age: [9,10],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-fluent-readers-male':
        {
         age: [11,12,13,14,15,16,17,18],
         sex: ['male'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-Black':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Black/African American'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-White':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['White/Caucasian'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-multiple-races':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Two or more races'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        'participants-reading_adhd2-Hispanic':
        {
         age: [int(t) for t in np.arange(6,22)],
         sex: ['male', 'female'],
          race: ['Hispanic'],
         dx_reading: ['reading_no_comorbidities', 'reading_adhd']
        },
        }
    
    # concat dicts
    spec_info = _concat_dicts(spec_info_adhd_all_comorbidities, 
                              spec_info_adhd_no_comorbidities, 
                              spec_info_adhd_other,
                              spec_info_depression_adhd,
                              spec_info_anxiety_adhd,
                              spec_info_gender_adhd_all_comorbidities, 
                              spec_info_gender_adhd_no_comorbidities,
                              spec_info_gender_other,
                              spec_info_gender_reading_all_comorbidities,
                              spec_info_gender_reading_no_comorbidities,
                              spec_info_gender_reading_adhd,
                              spec_info_reading_all_comorbidites, 
                              spec_info_reading_no_comorbidities,
                              spec_info_reading_adhd,
                              spec_info_reading_anxiety,
                              spec_info_adhd_no_reading,
                              spec_info_reading_anxiety2,
                              spec_info_reading_adhd2
                              )

    return base_info, spec_info


def targets():
    """hardcode target features
    """

    base_info = {}
    spec_info = {
        'target-Diagnosis-Reading':
                    {
                    'filename': 'participant_train_test.csv',
                    'target_column': 'DX_Reading', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'DX_Reading'], # columns we want in the final dataframe
                    },
        'target-Diagnosis-ADHD':
                    {
                    'filename': 'participant_train_test.csv',
                    'target_column': 'DX_ADHD', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'DX_ADHD'], # columns we want in the final dataframe
                    },
        'target-Diagnosis-Depression':
                    {
                    'filename': 'participant_train_test.csv',
                    'target_column': 'DX_Depression', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'DX_Depression'], # columns we want in the final dataframe
                    },
        'target-Diagnosis-Anxiety':
                    {
                    'filename': 'participant_train_test.csv',
                    'target_column': 'DX_Anxiety', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'DX_Anxiety'], # columns we want in the final dataframe
                    },
        'target-Diagnosis-ADHD-Subtype':
                    {
                    'filename': 'participant_train_test.csv',
                    'target_column': 'DX_Subtype_Name', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'DX_Subtype_Name'], # columns we want in the final dataframe
                    },
        'target-Gender': 
                    {
                    'filename': 'participant_train_test.csv',
                    'target_column': 'Sex', # should be string (e.g., 'DX_Cat_Name', Sex)
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'Sex'], # columns we want in the final dataframe
                    },
        'target-Continuous-Reading':
                    {
                    'filename': 'participant_train_test.csv',
                    'target_column': 'WIAT_RC_Raw', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': False,
                    'cols_to_keep': ['Identifiers', 'WIAT_RC_Raw'], # columns we want in the final dataframe
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
                "threshold": .9, # threshold dataframe based on some fixed criterion - remove features that are missing more than 10% of values
                }

    demos_info = {
            'features-demos':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season',  'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'Sex', 'Age_round',  'PreInt_Demos_Fam,Child_Race_cat', 'PreInt_Demos_Fam,Child_Ethnicity_cat']
            },
            'features-demos-excl-sex':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season',  'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'Age_round',  'PreInt_Demos_Fam,Child_Race_cat', 'PreInt_Demos_Fam,Child_Ethnicity_cat']
            }
            }

    # READING   
    reading_info = {
                'features-Child-language':
                {
                "filename": 'child-features-raw.csv',  # was `Child-Question.csv`
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CELF', 'PPVT', 'EVT', 'Sex', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-Child-phonological':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CTOPP', 'Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Child-phonological-minimal':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'EL_raw', 'BW_raw', 'NR_raw', 'RD_raw', 'RL_raw', 'RO_raw', 'Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Child-production':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'GFTA', 'Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Child-executive-function':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'NIH','Sex', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-Child-intelligence':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'WISC', 'WAIS', 'WAIS_Abb','KBIT','Sex','PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-Child-reading':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'WIAT', 'TOWRE','Sex','PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-Child-reading-minimal':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TOWRE,TOWRE_PDE_Raw','TOWRE,TOWRE_SWE_Raw','WIAT,WIAT_Word_Raw','WIAT,WIAT_RC_Raw', 'Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Child-emotional-status':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR', 'C3SR', 'SCARED_SR', 'CIS_SR', 'WHODAS_SR', 'PANAS','Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Parent-SES':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'Barratt', 'FSQ','Sex','PreInt_Demos_Fam,Child_Race_cat'] # cols related to SES
                },
                'features-Parent-Stress':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'PSI', 'DTS', 'APQ_P', 'PCIAT','Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Parent-Psychological-Function':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'NLES_P', 'PreInt_FamHx_RDC', 'Vineland', 'PreInt_Demos_Fam', 'PreInt_Demos_Home', 'PreInt_DevHx', 'PreInt_EduHx',  'PreInt_Lang', 'PreInt_TxHx','Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Parent-Intake-Interview':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers',  'PreInt_Demos_Fam', 'PreInt_Demos_Home', 'PreInt_DevHx', 'PreInt_EduHx',  'PreInt_Lang', 'PreInt_TxHx','Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Parent-Family-History':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'PreInt_FamHx_RDC','Sex','PreInt_Demos_Fam,Child_Race_cat']
                },
                'features-Child-reading-all':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CELF', 'PPVT', 'EVT', 'CTOPP', 'WIAT', 'TOWRE', 'Sex','PreInt_Demos_Fam,Child_Race_cat'] 
                }
                }

    # CBCL subscales
    internalizing_externalizing = {
                'features-parent-internalizing':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_Int', 'CBCL_Pre_Int', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
                },
                'features-parent-externalizing':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_Ext', 'CBCL_Pre_Ext', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
                },
                'features-teacher-internalizing':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_Int', 'TRF_P_Int', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],
                },
                'features-teacher-externalizing':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_Ext', 'TRF_P_Ext', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],
                },
                'features-child-internalizing':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_Int', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
                },
                'features-child-externalizing':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_Ext', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
                },
                }
    anxious_depressed = {
                'features-parent-anxious_depressed':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_AD', 'CBCL_Pre_AD', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-anxious_depressed':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_AD', 'TRF_AD', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-anxious_depressed':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_AD', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }
    withdrawn_depressed = {
                'features-parent-withdrawn_depressed':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_WD','CBCL_Pre_WD','PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-withdrawn_depressed':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_WD', 'TRF_P_WD', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-withdrawn_depressed':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_WD', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }
    social_problems = {
                'features-parent-social_problems':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_SP', 'CBCL_Pre_SP','PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-social_problems':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_SP', 'TRF_P_SP', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-social_problems':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_SP', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }
    thought_problems = {
                'features-parent-thought_problems':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_TP', 'CBCL_Pre_TP', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-thought_problems':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_TP', 'TRF_P_TP', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-thought_problems':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_TP', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }
    attention_problems = {
                'features-parent-attention_problems':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_AP', 'CBCL_Pre_AP', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-attention_problems':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_AP', 'TRF_P_AP', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-attention_problems':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_AP', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }
    rule_breaking = {
                'features-parent-rule_breaking':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_RBB', 'CBCL_Pre_RBB','PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-rule_breaking':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_RBB', 'TRF_P_RBB', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-rule_breaking':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_RBB', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }
    aggressive_behavior = {
                'features-parent-aggressive_behavior':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_AB', 'CBCL_Pre_AB','PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-aggressive_behavior':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_AB', 'TRF_P_AB', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-aggressive_behavior':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_AB', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }
    somatic_complaints = {
                'features-parent-somatic_complaints':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL_SC', 'CBCL_Pre_SC', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                'features-teacher-somatic_complaints':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF_SC', 'TRF_P_SC', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],# 'Sex'
                },
                'features-child-somatic_complaints':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR_SC', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],# 'Sex'
                },
                }

    # C3SR subscales
    C3SR_info = {
            'features-child-defiance_aggression':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'C3SR_AG', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
            'features-child-family_relations':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'C3SR_FR', 'PreInt_Demos_Fam,Child_Race_cat',  'Age_round'],
            },
            'features-child-hyperactive_impulsivity':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'C3SR_HY', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
            'features-child-inattention':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'C3SR_IN', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
            'features-child-learning_problems':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'C3SR_LP', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
            'features-child-negative_impression':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'C3SR_NI', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
            'features-child-positive_impression':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'C3SR_PI', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
            }

    # Suicide subscales
    CSSRS_info = {
            'features-child-suicidality':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'CSSRS', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
        }

    # SWAN subscales
    SWAN_info = {
            'features-child-swan_inattention':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'SWAN_IN', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
            'features-child-swan_hyperactive':
            {
            "filename": 'child-features-raw.csv', 
            "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
            "cols_to_filter": ['Identifiers', 'SWAN_HY', 'PreInt_Demos_Fam,Child_Race_cat', 'Age_round'],
            },
        }
    
    adhd_info = {     
                'features-all-questions':
                {
                "filename": 'all-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', '_Complete', '_Incomplete', 
                                 'START_DATE', 'Days_Baseline', 'Year', 'missing', 'present',
                                 '_Invalid', '_Valid'], # cols to drop from dataframe
                "cols_to_filter": None
                },      
                'features-child-connors':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'C3SR', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-child-cbcl':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR', 'ASR', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-child-anxiety':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SCARED_SR', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-child-mood':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'MFQ_SR', 'PANAS', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-child-suicide':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CSSRS', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-child-language-all':
                {
                "filename": 'child-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CELF_Full_5to8', 'CELF_Full_9to21', 'CELF_Meta', 'EVT', 'PPVT', 'GFTA', 'CTOPP', 'TOWRE', 'CELF', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-parent-cbcl':
                {
                "filename": 'parent-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL', 'CBCL_Pre', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-teacher-cbcl':
                {
                "filename": 'parent-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF', 'TRF_Pre', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-parent-anxiety':
                {
                "filename": 'parent-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SCARED_P', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                'features-parent-strengths-weaknesses-adhd-all':
                {
                "filename": 'parent-features-Question.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SWAN', 'ESWAN', 'SDQ', 'PreInt_Demos_Fam,Child_Race_cat']
                }, 
                }
    asd_info = {
                'features-parent-asd':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'ASSQ']
                }, 
                'features-parent-social-communication':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SCQ', 'SAS', 'SRS', 'SRS_Pre']
                }, 
                'features-parent-child-mind-institute':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'SympChck']
                }, 
                'features-parent-mood':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'MFQ_P']
                }
            }
    
    other = {
            'features-Child-CBCL':
                {
                "filename": 'child-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'YSR']
                },
            'features-Parent-CBCL':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'CBCL', 'CBCL_Pre']
                },
            'features-Teacher-CBCL':
                {
                "filename": 'teacher-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TRF', 'TRF_P']
                },
            'features-Reading-Raw':
                {
                "filename": 'child-features-raw.csv',
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'TOWRE', 'WIAT']
                },
            'features-SES':
                {
                "filename": 'parent-features-raw.csv', 
                "cols_to_drop": ['Administration', 'Data_entry', 'EID', 'Season', 'START_DATE', 'Study', 'Days_Baseline', 'Year', 'missing', 'present'], # cols to drop from dataframe
                "cols_to_filter": ['Identifiers', 'Barratt', 'FSQ'] 
                },
                }

    spec_info = _concat_dicts(demos_info, reading_info, adhd_info, C3SR_info, asd_info, internalizing_externalizing, 
                              anxious_depressed, withdrawn_depressed, social_problems, thought_problems, CSSRS_info,
                              SWAN_info, attention_problems, rule_breaking, aggressive_behavior, somatic_complaints,
                              other)
    
    return base_info, spec_info




