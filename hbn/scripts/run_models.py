import os
import numpy as np
import pandas as pd
import glob
import shutil
import click
from pydra_ml.classifier import gen_workflow, run_workflow

from hbn import io
from hbn.constants import Defaults

def test_classifier(tmpdir=Defaults.RAW_DIR):
    clfs = [
        ("sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}),
        [
            ["sklearn.impute", "SimpleImputer"],
            ["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.naive_bayes", "GaussianNB", {}],
        ],
    ]   
    # get features
    # csv_file = os.path.join(os.path.dirname(__file__), "data", "breast_cancer.csv")
    dataframe = build_features.phenotypes(
        assessment='Child Measures', 
        domains='Language Tasks',    
        diagnosis=True,
        CGAS_Score=False
        )
   
    csv_file = os.path.join(Defaults.FEATURE_DIR, 'Child_Measures-Language_Tasks.csv')
    dataframe.to_csv(csv_file)

    
    x_cols = dataframe.select_dtypes(include='number').columns.tolist()
    # x_cols = dataframe.columns.drop(['Identifiers', 'DX_01']).tolist()

    inputs = {
        "filename": csv_file,
        "x_indices": x_cols,
        "target_vars": ["DX_01"],
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [True, False],
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "accuracy",
        "gen_shap": True,
        "nsamples": 5,
        "l1_reg": "aic",
        "plot_top_n_shap": 16,
        "metrics": ["roc_auc_score", "accuracy_score"],
    }
    wf = gen_workflow(inputs, cache_dir=tmpdir)
    results = run_workflow(wf, "cf", {"n_procs": 1})
    assert results[0][0]["ml_wf.clf_info"][1] == "MLPClassifier"
    assert results[0][0]["ml_wf.permute"]
    assert results[0][1].output.score[0][0] < results[1][1].output.score[0][0]
    assert hasattr(results[2][1].output.model, "predict")
    assert isinstance(results[2][1].output.model.predict(np.ones((1, 30))), np.ndarray)


def test_regressor(tmpdir=Defaults.RAW_DIR):
    clfs = [
        [
            ["sklearn.impute", "SimpleImputer"],
            ["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.neural_network", "MLPRegressor", {"alpha": 1, "max_iter": 1000}],
        ],
        (
            "sklearn.linear_model",
            "LinearRegression",
            {"fit_intercept": True, "normalize": True},
        ),
    ]
    # csv_file = os.path.join(os.path.dirname(__file__), "data", "diabetes_table.csv")
    dataframe = build_features.phenotypes(
                assessment='Child Measures', 
                domains='Language Tasks',    
                diagnosis=False,
                CGAS_Score=True
                )
            
    csv_file = os.path.join(Defaults.FEATURE_DIR, 'Child_Measures-Language_Tasks.csv')
    dataframe.to_csv(csv_file, index=None)
    
    # x_cols = dataframe.select_dtypes(include='number').columns.tolist()
    x_cols = dataframe.columns.drop(['Identifiers', 'CGAS_Score']).tolist()

    inputs = {
        "filename": csv_file,
        "x_indices": x_cols,
        "target_vars": ["CGAS_Score"],
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [True, False],
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "accuracy",
        "gen_shap": True,
        "nsamples": 5,
        "l1_reg": "aic",
        "plot_top_n_shap": 10,
        "metrics": ["explained_variance_score"],
    }

    wf = gen_workflow(inputs, cache_dir=tmpdir)
    results = run_workflow(wf, "cf", {"n_procs": 1})
    assert results[0][0]["ml_wf.clf_info"][-1][1] == "MLPRegressor"
    assert results[0][0]["ml_wf.permute"]
    assert results[0][1].output.score[0][0] < results[1][1].output.score[0][0]
    assert hasattr(results[2][1].output.model, "predict")
    assert isinstance(results[2][1].output.model.predict(np.ones((1, 10))), np.ndarray)

# @click.command()
# @click.option("--spec_file")

def run(
    spec_file='regression-Child_Measures-Cognitive_Testing-all-CGAS_Score.json', 
    tmpdir='/Users/maedbhking/pydra-ml/cache-wf/'
    ):

    # load json
    spec_fpath = os.path.join(Defaults.BASE_DIR, "models", spec_file)
    spec_info = io.read_json(spec_fpath)

    # get features
    csv_file = os.path.join(Defaults.BASE_DIR, "features", spec_info['filename'])
    dataframe = pd.read_csv(csv_file)
    spec_info['filename'] = csv_file # full path to csv file

    spec_info['x_indices'] = range(1,len(dataframe.columns)-1)

    wf = gen_workflow(spec_info, cache_dir=tmpdir)
    results = run_workflow(wf, "cf", {"n_procs": 1})

    # move model output to new directory + add model spec file
    out_dir = glob.glob(os.path.join(os.getcwd(), '*out-localspec*'))
    shutil.copy(spec_fpath, out_dir[0])
    shutil.move(out_dir[0], Defaults.MODEL_DIR)
    shutil.rmtree("messages")

    # assert results[0][0]["ml_wf.clf_info"][-1][1] == "MLPRegressor"
    # assert results[0][0]["ml_wf.permute"]
    # assert results[0][1].output.score[0][0] < results[1][1].output.score[0][0]
    # assert hasattr(results[2][1].output.model, "predict")
    # assert isinstance(results[2][1].output.model.predict(np.ones((1, 10))), np.ndarray)

if __name__ == "__main__":
    run()