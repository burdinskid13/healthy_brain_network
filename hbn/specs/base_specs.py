import os
from hbn import io
from hbn.constants import Defaults
from hbn.features import build_features

def pydralml_base(n_splits=5, test_size=0.2):

    base_info = {
        "filename" : None,
        "x_indices" : None,
        "target_vars" : None,
        "permute" : [True, False],
        "group_var" : None,
        "n_splits" : n_splits,
        "test_size" : test_size,
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

    clf_info = {
        'spec1':
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
        ],
        'spec2':
        [
        [["sklearn.preprocessing", "StandardScaler"]
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]],
        ],
        'spec3':
        [
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]],
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "saga", "penalty": "l1"}]],
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50}]] 
        ],
        'spec4':
        [
            ["sklearn.feature_selection", "SelectFromModel", {"estimator": "LinearSVC"}],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]
        ]
        }
    return clf_info, base_info


def participant_base(out_dir=os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')):

    spec_info = {
        'spec-adhd-No_Diagnosis':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-All_Other_Diagnoses':
        {'diagnoses': ['ADHD', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-asd-All_Other_Diagnoses':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        # 'spec-adhd-Subtypes_02':
        # {'diagnoses': ['ADHD-Combined_Type', 'ADHD-Inattentive_Type'],
        #  'split': 'train',
        #  'age': 'all',
        #  'sex': 'all',
        #  'ethnicity': 'all'
        # },
        'spec-depression-No_Diagnosis':
        {'diagnoses': ['Depressive_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-depression-All_Other_Diagnoses':
        {'diagnoses': ['Depressive_Disorders', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-anxiety-No_Diagnosis':
        {'diagnoses': ['Anxiety_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-anxiety-All_Other_Diagnoses':
        {'diagnoses': ['Anxiety_Disorders', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-reading-No_Diagnosis':
        {'diagnoses': ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-reading-All_Other_Diagnoses':
        {'diagnoses': ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-No_Diagnosis-male':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-All_Other_Diagnoses-male':
        {'diagnoses': ['ADHD', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-No_Diagnosis-female':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-All_Other_Diagnoses-female':
        {'diagnoses': ['ADHD', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis-male':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-asd-All_Other_Diagnoses-male':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis-female':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-asd-All_Other_Diagnoses-female':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'All_Other_Diagnoses'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-reading-No_Diagnosis-male':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-All_Other_Diagnoses-male':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'All_Other_Diagnoses'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-No_Diagnosis-female':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-reading-All_Other_Diagnoses-female':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'All_Other_Diagnoses'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-No_Diagnosis-female':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-All_Other_Diagnoses-female':
        {
        "diagnoses": ['Anxiety_Disorders', 'All_Other_Diagnoses'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-No_Diagnosis-male':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-anxiety-All_Other_Diagnoses-male':
        {
        "diagnoses": ['Anxiety_Disorders', 'All_Other_Diagnoses'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-depression-No_Diagnosis-female':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-All_Other_Diagnoses-female':
        {
        "diagnoses": ['Anxiety_Disorders', 'All_Other_Diagnoses'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-depression-No_Diagnosis-male':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-depression-All_Other_Diagnoses-male':
        {
        "diagnoses": ['Depressive_Disorders', 'All_Other_Diagnoses'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-age-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-05_06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-07_08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7,8],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-09_10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9,10],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-11_12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11,12],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-13_14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-15_16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-05_06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-07_08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7,8],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-09_10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9,10],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-11_12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11,12],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-13_14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-15_16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-05_06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-07_08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7,8],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-09_10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9,10],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-11_12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11,12],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-13_14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-15_16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-17':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-anxiety':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-depression':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-asd':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-reading':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-anxiety-male':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-depression-male':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-asd-male':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-reading-male':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-anxiety-female':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-depression-female':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-asd-female':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-reading-female':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-reading-anxiety':
        {
        "diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-reading-depression':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-reading-asd':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-reading-No_Diagnosis':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "No_Diagnosis_Given"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-reading-anxiety-male':
        {
        "diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-depression-male':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-asd-male':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-No_Diagnosis-male':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "No_Diagnosis_Given"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-anxiety-female':
        {
        "diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-reading-depression-female':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-reading-asd-female':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-reading-No_Diagnosis-female':
        {"diagnoses": [
            "Specific_Learning_Disorder_with_Impairment_in_Reading",
            "No_Diagnosis_Given"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-multilabel':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder",
            "Anxiety_Disorders",
            "Depressive_Disorders",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        }
    return spec_info


def parent_features_base(out_dir=Defaults.FEATURE_DIR):

    base_info = {
                "preprocessing": {
                    "preprocess": True,
                    "cols_to_drop": ['EID', 'Comment_ID', 'Unnamed', 'Administration', 'Days_Baseline', 'Data_entry', 'START_DATE', 'Year', 'Site', 'Season', 'Visit_label', 'Study', 'PSCID', 'Release_Number'], # cols to drop while preprocessing
                    "cols_to_ignore": ['Identifiers'], # cols to ignore in the preprocessing routine (column transformation)
                    "upsample": True, # upsample minority class using SMOTE
                    "threshold": False, #threshold dataframe based on some fixed criterion
                    "clf_info": {
                        "numeric": [
                            [
                                "sklearn.impute",
                                "SimpleImputer",
                                {
                                    "strategy": "mean"
                                }
                            ],
                            [
                                "sklearn.preprocessing",
                                "StandardScaler",
                                {}
                            ]
                        ]
                        }
            }
            }

    return base_info


def targets():
    """hardcode target features
    """
    target_info = [
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "binarize",
                    "outname": "DX_01_Cat_binarize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat_new",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "binarize",
                    "outname": "DX_01_Cat_new_binarize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat_new",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "factorize",
                    "outname": "DX_01_Cat_new_factorize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "factorize",
                    "outname": "DX_01_Cat_factorize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01",
                    "features_to_ignore": ['KSADS','Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "binarize",
                    "outname": "DX_01_binarize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01",
                    "features_to_ignore": ['KSADS','Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "factorize",
                    "outname": "DX_01_factorize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "Sex",
                    "features_to_ignore": ['KSADS','Sex'],
                    "transform": "binarize",
                    "outname": "Sex_binarize"
                    }
                ]

    return target_info


def features():
    # get separate bases - we will get all combinations of assessment*domains*measures to create unique feature specs
    feature_info = {
            'basic_demographics':
            {"assessment": ["Child Measures", "Parent Measures", "Teacher Measures", "Clinical Measures"],
            "domains": "all",
            "measures": "all",
            "abbrevs": 'all',
            "filter_features": {'filename': None, 'columns': None},
            "add_features": {'filename': 'Demographic_Features.csv', 'columns': ['Sex', 'Age', 'Diagnosis', 'Category', 'comorbidities', 'Race', 'Ethnicity']} 
            }, 
            'total_scores_demographics':
            {"assessment": ["Child Measures", "Parent Measures", "Teacher Measures", "Clinical Measures"],
            "domains": "all",
            "measures": "all",
            "abbrevs": 'all',
            "filter_features": {'filename': 'item-names-cleaned.csv', 'columns': ['Total_Scores']},
            "add_features": {'filename': 'Demographic_Features.csv', 'columns': ['Sex', 'Age', 'Diagnosis', 'Category', 'comorbidities', 'Race', 'Ethnicity']}   
            },
            'remove_total_scores_demographics':
            {"assessment": ["Child Measures", "Parent Measures", "Teacher Measures", "Clinical Measures"],
            "domains": "all",
            "measures": "all",
            "abbrevs": 'all',
            "filter_features": {'filename': 'item-names-cleaned.csv', 'columns': ['Not_Total_Scores']},
            "add_features": {'filename': 'Demographic_Features.csv', 'columns': ['Sex', 'Age', 'Diagnosis', 'Category', 'comorbidities', 'Race', 'Ethnicity']}   
            },
            'free_assessments_demographics':
            {"assessment": ["Child Measures", "Parent Measures", "Teacher Measures", "Clinical Measures"],
            "domains": "all",
            "measures": "all",
            "abbrevs": 'all',
            "filter_features": {'filename': 'item-names-cleaned.csv', 'columns': ['Free_Assessments']},
            "add_features": {'filename': 'Demographic_Features.csv', 'columns': ['Sex', 'Age', 'Diagnosis', 'Category', 'comorbidities', 'Race', 'Ethnicity']}   
            },
            'proprietary_assessments_demographics':
            {"assessment": ["Child Measures", "Parent Measures", "Teacher Measures", "Clinical Measures"],
            "domains": "all",
            "measures": "all",
            "abbrevs": 'all',
            "filter_features": {'filename': 'item-names-cleaned.csv', 'columns': ['Proprietary_Assessments']},
            "add_features": {'filename': 'Demographic_Features.csv', 'columns': ['Sex', 'Age', 'Diagnosis', 'Category', 'comorbidities', 'Race', 'Ethnicity']}   
            }
    }

    # loop over these feature sets and make unique feature specs
    spec_info = {}
    for k,v in feature_info.items():

        # get all combinations
        specs = _get_feature_combinations(spec_info=v)
        spec_info.update({k: specs})

    return spec_info


def _get_feature_combinations(spec_info):
    """gets combinations of assessment*domain*measure to make many feature files from `parent_spec`

    horrible code -- need to rewrite

    Args:
        parent_spec (dict): parent spec info (output from `make_parent_spec`)
    """

    assessments = spec_info['assessment']
    domains = spec_info['domains']
    measures = spec_info['measures']
    abbrevs = spec_info['abbrevs']
    filter_features = spec_info['filter_features']
    add_features = spec_info['add_features']

    # check arguments
    if not isinstance(assessments, list):
        assessments = [assessments]
    if (not isinstance(domains, list) and (domains!='all')):
        domains = [domains]
    if (not isinstance(measures, list) and (measures!='all')):
        measures = [measures]

    # write out all possible feature combinations
    spec_info = []
    for assess in assessments:
        if domains=='all':
            domain_names = build_features.get_domains(assess)[assess]
            if domain_names is not None:
                domain_names.remove('all')
            elif domain_names is None:
                domain_names = [domain_names]
        for domain in domain_names:
            if measures=='all':
                measure_names = build_features.get_measures(assess, domain)[domain]
            for measure in measure_names:
                abbrevs = build_features.get_abbrevs(assess, measure)
                for abbrev in abbrevs:
                    spec_info.append({'assessment': assess,
                            'domains': domain,
                            'measures': measure,
                            'abbrevs': abbrev,
                            'filter_features': filter_features,
                            'add_features': add_features
                            })

        # write out assessment-feature models
        spec_info.append(
            {'assessment': assess,
            'domains': 'all',
            'measures': 'all',
            'abbrevs': 'all',
            'filter_features': filter_features,
            'add_features': add_features
            })

        # write out all-feature models
        spec_info.append(
            {'assessment': 'all',
             'domains': 'all',
             'measures': 'all',
             'abbrevs': 'all',
             'filter_features': filter_features,
             'add_features': add_features
             })

    return spec_info
