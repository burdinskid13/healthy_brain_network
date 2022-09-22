

def classifier_umap(X, y):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    from umap import UMAP

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Classification with a linear SVM
    svc = LinearSVC(dual=False, random_state=123)
    params_grid = {"C": [10 ** k for k in range(-3, 4)]}
    clf = GridSearchCV(svc, params_grid)
    clf.fit(X_train, y_train)
    print(
        "Accuracy on the test set with raw data: {:.3f}".format(clf.score(X_test, y_test))
    )

    # Transformation with UMAP followed by classification with a linear SVM
    umap = UMAP(random_state=456)
    pipeline = Pipeline([("umap", umap), ("svc", svc)])
    params_grid_pipeline = {
        "umap__n_neighbors": [5, 20],
        "umap__n_components": [15, 25, 50],
        "svc__C": [10 ** k for k in range(-3, 4)],
    }


    clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline)
    clf_pipeline.fit(X_train, y_train)
    print(
        "Accuracy on the test set with UMAP transformation: {:.3f}".format(
            clf_pipeline.score(X_test, y_test)
        )
    )

def load_data(
    assessment='Child Measures', 
    domains='all', 
    target='Sex_binarize'
    ):
    from hbn.features import build_features

    # get data
    df = build_features.get_data(assessment=assessment, domains=domains, target=target, min_num_participants=4000)

    # get preprocessing classifier info
    clf_info = {"numeric": [["sklearn.impute", "SimpleImputer", {"strategy": "mean"}], ["sklearn.preprocessing", "StandardScaler", {}]]}

    # prprocess data
    df = build_features.preprocess(dataframe=df, clf_info=clf_info, cols_to_ignore=[target])

    return df.drop([target], axis=1), df[target]