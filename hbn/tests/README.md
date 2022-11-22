Tests
==============================

### Test workflow
```
# navigate to tests directory
cd ../hbn/tests

# run test workflow
python3 test_workflow.py --cachedir=/home/shreyark/.cache/pydra-ml/cache-wf/
```

### Model Specs
* **classifier-Extended_Strengths_and_Weaknesses_Assessment_of_Normal_Behavior-Parent_Report-DX_01_Cat_binarize-733973276-spec.json** is a model spec file
* The spec file contains all of the information needed to run the model that is input to **hbn.models.first_level_modeling.run_pipeline**
> Note: model spec files can be created programatically with ** hbn.models.first_level_modeling.make_model_spec**

### Example Model Spec
For example: the following parameters contain the following information:
> For a more detailed description of the parameters, see https://github.com/nipype/pydra-ml
* filename: the __exact__ filename of the .csv features file to be input to modeling pipeline (NOT full path)
* x_indices: can be left empty as the X columns will be input programmatically in **hbn.models.first_level_modeling.run_pipeline**
* target_vars: should be a list containing one target variable of the following: "DX_01_Cat_binarize", "DX_01_Cat_factorize", "Sex_binarize"
* additional parameters are included in the model spec for the purposes of preserving information about `filename`

Example Model Spec
----------------------
```
{
    "filename": "model_features_737819522.csv",
    "feature_spec": {
        "assessment": "Parent Measures",
        "domains": "Demographic Questionnaire Measures",
        "measures": "Extended Strengths and Weaknesses Assessment of Normal Behavior-Parent Report",
        "preprocessing": {
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
        },
        "min_num_participants": 2000
    },
    "target_spec": {
        "assessment": "Clinical Measures",
        "domain": null,
        "measure": "Clinical Diagnosis Demographics",
        "target_column": "DX_01_Cat",
        "transform": "binarize",
        "outname": "DX_01_Cat_binarize",
        "preprocessing": {
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
        },
        "min_num_participants": 2000
    },
    "participants": [
        "train_participants-ADHD.csv",
        "train_participants-No_Diagnosis_Given.csv"
    ],
    "x_indices": [],
    "target_vars": [
        "DX_01_Cat_binarize"
    ],
    "group_var": null,
    "n_splits": 50,
    "test_size": 0.2,
    "clf_info": [
        [
            "sklearn.tree",
            "DecisionTreeClassifier",
            {
                "max_depth": 5
            }
        ]
    ],
    "permute": [
        true,
        false
    ],
    "gen_feature_importance": true,
    "gen_permutation_importance": false,
    "permutation_importance_n_repeats": 5,
    "permutation_importance_scoring": "accuracy",
    "gen_shap": false,
    "nsamples": "auto",
    "l1_reg": "aic",
    "plot_top_n_shap": 10,
    "metrics": [
        "roc_auc_score",
        "f1_score",
        "precision_score",
        "recall_score"
    ]
}
```
