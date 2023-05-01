from hbn.constants import Defaults


def pydraml_base(clf_info, n_splits=5, test_size=0.2):
    spec_info = {
    "filename" : None,
    "x_indices" : None,
    "target_vars" : None,
    "clf_info" : clf_info,
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

    return spec_info


def make_specs(out_dir=Defaults.MODEL_SPEC_DIR, n_splits=5, test_size=0.2):
    import os
    from hbn import io

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
        ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]
        ]
        }

    # loop over classifies and save out pydra-ml specs
    for name,clf in clf_info.items():

        # create spec parameters
        spec_info = pydraml_base(clf_info=clf, n_splits=n_splits, test_size=test_size)

        # write out pydra-ml specs
        fpath = os.path.join(out_dir, f'pydraml_{name}.json')
        io.save_dict_as_JSON(fpath, spec_info)

        

            

