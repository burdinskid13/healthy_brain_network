
def calculate_embedding(
    data, 
    standarize=True,
    n_neighbors=15, 
    min_dist=0.1, 
    n_components=2,
    metric='euclidean', 
    ): 
    """ calculate embedding using UMAP
    
    Args:
        data (pd dataframe): shape (n_rows, n_cols)
        standarize (bool): whether to standarize data
        n_neighbors (int): number of neighbors
        min_dist (float): minimum distance
        n_components (int): number of components
        metric (str): metric to use for UMAP
    Returns:
        embedding (pd dataframe): shape (n_rows, n_components)
    """
    import umap
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    # construct model
    reducer = umap.UMAP(
        random_state=42, 
        n_components=n_components, 
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        metric=metric
        )
    
    # impute if there are NaN
    imputer = SimpleImputer(strategy='mean')  # Replace 'mean' with 'median', 'most_frequent', etc.
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # optionally standarize
    if standarize:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    # train model 
    reducer.fit(data)
    
    # to access the resulting transform, we inspect `embedding_` attribute
    # or call transform on the original data
    embedding = reducer.transform(data)
#     embedding = reducer.fit_transform(data);

    # rename embeddings as 'x', 'y', 'z'
    columns = ['x', 'y']
    if n_components==3:
        columns = ['x', 'y', 'z']

    # rename embeddings 
    embedding = pd.DataFrame(embedding, columns=columns)

    return embedding


def plot_umap_2d(embedding, target=None):
    from matplotlib import pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd

    # check if embedding is a pd dataframe
    if not isinstance(embedding, pd.DataFrame):
        # return error if not
        raise ValueError('embedding must be a pd dataframe')
    
    if target is not None:
        target = embedding[target]

    # now plot the resulting embedding, coloring the data points by the class that they belong to
    fig, ax = plt.subplots(figsize=(5, 5))
    ax = sns.scatterplot(data=embedding, x="x", y="y", hue=target, ax=ax)

    # Set the legend title to an empty string
    ax.legend().set_title("")

    # set aspect ratio
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()


def plot_umap_3d(embedding, target=None):
    import plotly.express as px
    import pandas as pd

    # # visualize embedding (3d display)
    # df_embedding = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
    # dataframe = pd.concat([df_embedding, target], axis=1)

    # check if embedding is a pd dataframe
    if not isinstance(embedding, pd.DataFrame):
        # return error if not
        raise ValueError('embedding must be a pd dataframe')


    fig = px.scatter_3d(embedding, x='x', y='y', z='z', color=target) 
    fig.show()
