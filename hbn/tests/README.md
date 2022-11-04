Tests
==============================

### Feature Specs
* **features-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize-spec.json** is a feature spec file (.json)
* The spec file contains all of the information needed to create the features file **features-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize.csv**

You will need to create the .csv file using the feature spec file.
Run the following command
```
import os
from hbn.constants import Defaults
from hbn.features import build_features

TEST_DATA = os.path.join(Defaults.TEST_DIR, 'test_data')

# define feature spec file
feature_spec = os.path.join(TEST_DATA, 'features-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize-spec.json')

# make features csv
build_features.make_feature_files(feature_spec, out_dir=TEST_DATA)

```

### Example Feature Spec
The following parameters contain the following information:
* filename: the exact filename (.csv) that the features will be saved to
* features_X:
    * assessment: should always be one of the following: "Child Measures", "Parent Measures", "Teacher Measures"
    * domains: to get all possible __domains__ for each __assessment__, run **hbn.features.build_features.get_domains**
    * measures: to get all possible __measures__ for each __domains__ (and __assessment__), run **hbn.features.build_features.get_measures**
* target_y:
    * assessment: assessment in which `domain` and `measure` are saved
    * domain: domain in which `measure` is saved 
    * measure: questionnaire name (e.g., `Clinical Diagnosis Demographics`)
    * target_column: any column from `measure`
    * transform: should be one of the following - "binarize", "factorize", "numeric"
    * outname: should be one of the following - "DX_01_Cat_binarize", "DX_01_Cat_factorize", "Sex_binarize"
* preprocessing: follow example given in **Example Feature Spec** below
* min_num_participants: should always be an integer value

Example Feature Spec
----------------------
```
{
    "filename": "features-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize.csv",
    "features_X": {
        "assessment": "Parent Measures",
        "domains": "Interview of Emotional and Psychological Function",
        "measures": "Intake Interview"
    },
    "target_y": {
        "assessment": "Clinical Measures",
        "domain": null,
        "measure": "Clinical Diagnosis Demographics",
        "target_column": "DX_01_Cat",
        "transform": "binarize",
        "outname": "DX_01_Cat_binarize"
    },
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
        ],
        "category": [
            [
                "sklearn.impute",
                "SimpleImputer",
                {
                    "strategy": "most_frequent"
                }
            ],
            [
                "sklearn.preprocessing",
                "OneHotEncoder",
                {
                    "handle_unknown": "ignore",
                    "sparse": false
                }
            ]
        ]
    },
    "min_num_participants": 2000
}
```

### Model Specs
* **classifier-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize.json** is a model spec file (.json)
* The spec file contains all of the information needed to create the model that is input to **hbn.models.first_level_modeling.run_pipeline**
> Note: model spec files can be created programatically with ** hbn.models.first_level_modeling.make_specs**

You will need to create the model spec file using the feature spec file.
Run the following command
```
import os
from hbn.constants import Defaults
from hbn.models import first_level_modeling as first_level

TEST_DATA = os.path.join(Defaults.TEST_DIR, 'test_data')

# define feature spec file
feature_spec = os.path.join(TEST_DATA, 'features-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize-spec.json')

# make model spec file
model_spec = first_level.make_model_spec(feature_spec, participants='train_participants-ADHD.csv', out_dir=TEST_DATA)
```

### Example Model Spec
For example: the following parameters contain the following information:
> For a more detailed description of the parameters, see https://github.com/nipype/pydra-ml
* filename: the __exact__ filename of the .csv features file to be input to modeling pipeline
* x_indices: can be left empty as the X columns will be input programmatically in **hbn.models.first_level_modeling.run_pipeline**
* target_vars: should be a list containing one target variable of the following: "DX_01_Cat_binarize", "DX_01_Cat_factorize", "Sex_binarize"

Example Model Spec
----------------------
```
{
    "filename": "features-Parent_Measures-Interview_of_Emotional_and_Psychological_Function-Intake_Interview-DX_01_Cat_binarize.csv",
    "participants": "train_participants-ADHD.csv",
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
    "gen_permutation_importance": true,
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
