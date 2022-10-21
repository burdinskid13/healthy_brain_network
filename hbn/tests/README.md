Tests
==============================


### Feature Specs
* **features-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize-spec.json** is a feature spec file (.json)
* The spec file contains all of the information needed to create the features file **features-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize.csv**

The following parameters contain the following information:
* filename: the exact filename (.csv) that the features will be saved to
* assessment: should always be one of the following: "Child Measures", "Parent Measures", "Teacher Measures"
* domains: to get all possible __domains__ for each __assessment__, run **hbn.features.build_features.get_domains**
* measures: to get all possible __measures__ for each __domains__ (and __assessment__), run **hbn.features.build_features.get_measures**
* target: targets should be one of the following: "DX_01_Cat_binarize", "DX_01_Cat_factorize", "Sex_binarize"
* target_type: should always be "categorical" unless the __target__ is a continuous variable
* preprocessing: follow example given in **features-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize-spec.json**
* min_num_participants: should always be an integer value

Example Feature Spec
----------------------
```
{
    "filename": "features-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize.csv",
    "assessment": "Child Measures",
    "domains": "Cognitive Testing",
    "measures": "all",
    "target": "DX_01_Cat_binarize",
    "target_type": "categorical",
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
    "min_num_participants": 4000
}
```

### Model Specs
* **classifier-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize.json** is a model spec file (.json)
* The spec file contains all of the information needed to create the model that is input to **hbn.models.first_level_modeling.run_pipeline**
> Note: model spec files can be created programatically with ** hbn.models.first_level_modeling.make_specs**

For example: the following parameters contain the following information:
> For a more detailed description of the parameters, see https://github.com/nipype/pydra-ml
* filename: the __exact__ filename of the .csv features file to be input to modeling pipeline
* x_indices: can be left empty as the X columns will be input programmatically in **hbn.models.first_level_modeling.run_pipeline**
* target_vars: should be a list containing one target variable of the following: "DX_01_Cat_binarize", "DX_01_Cat_factorize", "Sex_binarize"

Example Model Spec
----------------------
```
{
    "filename": "features-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize.csv",
    "x_indices": [],
    "target_vars": [
        "DX_01_Cat_binarize"
    ],
    "group_var": null,
    "n_splits": 15,
    "test_size": 0.2,
    "clf_info": [
        [
            [
                "sklearn.impute",
                "SimpleImputer"
            ],
            [
                "sklearn.preprocessing",
                "StandardScaler"
            ],
            [
                "sklearn.tree",
                "DecisionTreeClassifier",
                {
                    "max_depth": 5
                }
            ]
        ]
    ],
    "permute": [
        true,
        false
    ],
    "gen_feature_importance": true,
    "gen_permutation_importance": false,
    "permutation_importance_n_repeats": 5,
    "permutation_importance_scoring": null,
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
