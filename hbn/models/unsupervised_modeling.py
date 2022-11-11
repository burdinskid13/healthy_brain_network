
def compute_svd(dataframe):
    """This function returns the output from svd and pcs.
    Args:
        dataframe: given by return_grouped_data
    """
    import numpy as np
    import pandas as pd
    # do svd
    u, s, vt = np.linalg.svd(dataframe, full_matrices=False)
    # reconstructed_data = pd.DataFrame(u[:, 0:k] @ np.diag(s[0:k]) @ vt[0:k, :], columns = dataframe.columns)

    # get the pcs (P=XV or P=US)
    # pcs = u * s 
    pcs = dataframe @ vt.T

    # get the first n pcs
    pcs= pd.DataFrame(pcs).add_prefix('pc')
    
    return u, s, vt, pcs


def variance_explained(dataframe, pcs=[1]):
    """ Gets variance explained for n pcs
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            pcs (list of int): which pc to include in variance explained. 
    """
    import numpy as np

    _, s, _, _ = compute_svd(dataframe)

    var_all = (s**2)/np.sum(s**2)

    # zero index
    pcs = [x-1 for x in pcs]
    pcs_var_fraction = np.sum(var_all[pcs])

    return pcs_var_fraction


def _reorder_dendrogram_leaves(dataframe):
    import numpy as np
    from scipy.spatial.distance import cdist
    
    # calculate euclidean distances between regions
    Y = cdist(dataframe.T, dataframe.T, 'euclidean')
    Y[Y == 0] = 'nan'

    n_regions = len(dataframe.columns) - 1
    distances = []
    for i in np.arange(n_regions):
        distances.append(np.nanmean(Y[0] - np.nanmean(Y[i+1])))

    # get indices of distances
    idx = np.argsort(distances)
    idx = list(idx+1)
    idx.insert(0,0)

    # reorder dataframe using sorted indices
    cols = dataframe.columns.tolist()
    cols_reorder = []
    for i in idx:
        cols_reorder.append(cols[i])
    
    return dataframe[cols_reorder]


def dendrogram_plot(
    dataframe, 
    method='ward', 
    metric='euclidean', 
    reorder=True, 
    orientation='top', 
    color_leaves=True, 
    ax=None, 
    **kwargs
    ):
    """ Plots dendrogram plot.
        
        Args:
            dataframe: 
            method (str): 'ward' # there are other options given by linkage
            metric (str): 'euclidean' # there are other options given by linkage
            ax (bool): figure axes. Default is None
            color_leaves (bool): whether or not leaf nodes should be colored
            orientation (str): how the dendrogram should be oriented. Default is 'top'. Other options are 'right', 'left'
            reorder (bool): reorders the expression matrix before being input to clustering algo. See Gomez et al. (2019) for details
                kwargs (dict): dictionary of additional (optional) kwargs.
                may include any graphical argument relevant to scipy's dendrogram
    """
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette

    if ax is None:
        plt.figure(num=1, figsize=[25,8])

    if reorder:
        dataframe = _reorder_dendrogram_leaves(dataframe)

    if color_leaves:
        set_link_color_palette(['b', 'r', 'y', 'm'])
    else:
        set_link_color_palette(['k', 'k', 'k', 'k'])

    R = dendrogram(
        Z=linkage(dataframe, method, metric),
        orientation=orientation,
        get_leaves=True,
        color_threshold=35.0,
        labels=dataframe.index.to_list(),
        distance_sort='ascending',
        above_threshold_color='black', 
        ax=ax, 
        **kwargs
        )

    # plt.title("Hierarchical Clustering Dendrogram", fontsize=20)
    plt.xlabel('')
    plt.ylabel(f"{metric.capitalize()} Distance")

    return R


def scree_plot(dataframe, ax=None):
    """ Plots scree plot for n pcs
        
        Args:
            dataframe: 
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn's lineplot
    """
    from matplotlib import pyplot as plt
    import seaborn as sns
    import numpy as np

    u, s, vt, pcs = compute_svd(dataframe)

    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    ax = sns.lineplot(x=np.arange(len(s)), y=s**2 / sum(s**2), ax=ax)
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Variance Explained')
    # ax.set_title('Scree Plot')
    plt.show()


def k_means_n_dims_heatmap(dataframe, num_clusters=2, ax=None):
    """ Plots n-dim k means heatmap
        
        Args:
            dataframe:
            num_clusters (int): number of clusters for k-means. Default is 2
            ax (bool): figure axes. Default is None
    """
    from sklearn import cluster
    import plotly.graph_objs as go
    import plotly.offline as py
    from matplotlib import pyplot as plt
    
    clusterer = cluster.KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_labels = clusterer.fit_predict(dataframe)

    dataframe['km_labels'] = kmeans_labels

    # split df into cluster groups
    grouped = dataframe.groupby(['km_labels'], sort=True)

    # compute sums for every column in every group
    df_n_dims = grouped.sum()

    # get x and y labels
    x_labels = df_n_dims.columns
    y_labels = []
    for y in df_n_dims.index.values:
        y_labels.append('cluster' + str(y+1))

    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    # visualise heatmap
    data = [go.Heatmap(z=df_n_dims.values.tolist(), 
                       y=y_labels,
                       x=x_labels,
                       colorscale='Viridis')]
    py.iplot(data, filename='pandas-heatmap')