

def calculate_embedding(
    data, 
    n_neighbors=15, 
    min_dist=0.1, 
    n_components=2,
    metric='euclidean', 
    ): 
    import umap
    
    # construct model
    reducer = umap.UMAP(
        random_state=42, 
        n_components=n_components, 
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        metric=metric
        )
    
    # train model 
    reducer.fit(data)
    
    # to access the resulting transform, we inspect `embedding_` attribute
    # or call transform on the original data
    embedding = reducer.transform(data)
#     embedding = reducer.fit_transform(data);

    return embedding


def visualize_embedding(embedding, target):
    """visualize embedding
    
    Args: 
        features (pd dataframe): shape (n_rows, n_cols) 
        target (pd dataframe): shape (n_rows,1) same n_rows as `features`
    """
    import pandas as pd


    # visualize embedding (3d display)
    df_embedding = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
    dataframe = pd.concat([df_embedding, target], axis=1)

    plot_umap_3d(dataframe, hue=target)


def plot_umap(embedding, target=None, title=''):
    from matplotlib import pyplot as plt
    import numpy as np
    
    # now plot the resulting embedding, coloring the data points by the class that they belong to
    n_components = embedding.shape[1]
    fig = plt.figure()
    if n_components==1:
        fig.add_subplot(111)
        plt.scatter(embedding[:, 0], range(len(embedding)), c=target, cmap='Spectral', s=5)
    if n_components==2:
        fig.add_subplot(111)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=5)
    if n_components==3:
        fig.add_subplot(111, projection='3d')
        plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=5)
    
    plt.title(title, fontsize=18)
    plt.gca().set_aspect('equal', 'datalim')
    if target is not None:
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    
    plt.show()


def plot_umap_3d(dataframe, hue=None):
    import plotly.express as px

    fig = px.scatter_3d(dataframe, x='x', y='y', z='z', color=hue)
    fig.show()
