import os

import numpy as np

from pydra_ml.classifier import gen_workflow, run_workflow


def test_classifier(tmpdir):
    clfs = [
        [["sklearn.preprocessing", "StandardScaler"],
        ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50}]]
        ], # classifier has to be last list
    # csv_file = os.path.join(os.path.dirname(__file__), "data", "breast_cancer.csv")
    csv_file = os.path.join('/om2/user/maedbh/pydra-ml/pydra_ml/tests/data', 'breast_cancer.csv')
    inputs = {
        "filename": csv_file,
        "x_indices": range(10),
        "target_vars": ["target"],
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [True, False],
        "oversample": True,
        "feature_selection": True,
        "feature_selection_strategy": 'intersection', # 'intersection or 'union'
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "accuracy",
        "gen_shap": True,
        "nsamples": 15,
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
    assert isinstance(results[2][1].output.model.predict(np.ones((1, 10))), np.ndarray)


if __name__ == "__main__":
    test_classifier(tmpdir='/om2/user/maedbh/bin/.cache')
