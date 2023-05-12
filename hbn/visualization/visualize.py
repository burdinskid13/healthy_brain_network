import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plotting_style():
    plt.style.use('seaborn-poster') # ggplot
    params = {'axes.labelsize': 30,
            'axes.titlesize': 25,
            'legend.fontsize': 25,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            # 'figure.figsize': (10,5),
            'font.weight': 'regular',
            # 'font.size': 'regular',
            'font.family': 'sans-serif',
            'lines.markersize': 20,
            'font.serif': 'Helvetica Neue',
            'lines.linewidth': 4,
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False}
    plt.rcParams.update(params)
    sns.set_context(rc={'lines.markeredgewidth': 0.1})
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})


def wordcloud(dataframe, column):
    """print a word cloud from `column` of a `dataframe`

    Args:
        dataframe (pd dataframe):
        column (str): column must be in dataframe

    """
    from wordcloud import WordCloud, STOPWORDS

    comment_words = ''
    stopwords = set(STOPWORDS)
    
    # iterate through the csv file
    for val in dataframe[column]:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "

    comment_words = comment_words.replace("nan", "")
    
    wordcloud = WordCloud(width=800, height=800,
                    background_color='white',
                    stopwords=stopwords,
                    min_font_size=10).generate(comment_words)
    
    # plot the WordCloud image                      
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    plt.show()


def umap_embeddings(dataframe, target):
    """compute and visualize umap embeddings for `dataframe` and `target`

    Args: 
        dataframe (pd dataframe): dataframe must contain `target`
        target (str): determines which target we will use to color the resulting embedding
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    # define umap object
    reducer = umap.UMAP()

    # get data
    data = dataframe[dataframe.columns].values

    # data should be standarized
    scaled_data = StandardScaler().fit_transform(data)

    # train reducer, learning the manifold to get reduced representations
    embedding = reducer.fit_transform(data)

    # check `target` type
    dataframe[target] = dataframe[target].astype(int)

    plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in dataframe[target]]
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection', fontsize=24)
    plt.show()


def predictive_modeling(df, x='features', y='roc_auc_score'):
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    
    fig = go.Figure()

    fig.add_trace(go.Violin(x=df[x][df['data']=='data'],
                            y=df[y][df['data']=='data'],
                            legendgroup='data', scalegroup='data', name='data',
                            side='negative',
                            line_color='blue')
                 )
    fig.add_trace(go.Violin(x=df[x][df['data']=='null'],
                            y=df[y][df['data']=='null'],
                            legendgroup='null', scalegroup='null', name='null',
                            side='positive',
                            line_color='orange')
                 )
    fig.add_hline(y=.5, line_width=1, line_dash="dash", line_color="black")
    fig.update_traces(meanline_visible=True, box_visible=False)
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(title_text=y, range=[0.4, 1])
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                  'paper_bgcolor': 'rgba(0,0,0,0)'})
    fig.show()


def predictive_modeling_group(df, x='participant_group', y='roc_auc_score', title=None):
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import plotly as px
    
    fig = go.Figure()

    line_colors = px.colors.sequential.Plasma_r
    for group, color in zip(df[x].unique(), line_colors):

        df1 = df[(df[x]==group) & (df['data']=="data")]

        fig.add_trace(go.Violin(x=df1[x][df1[x]==group],
                                y=df1[y][df1[x]==group],
                                name=group,
                                line_color=color
                                )
                    )
        fig.add_hline(y=.5, line_width=1, line_dash="dash", line_color="black")

    fig.update_traces(meanline_visible=True, box_visible=False)
    fig.update_layout(violingap=0, violinmode='overlay', title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(title_text='ROC AUC', range=[0.4, 1])
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                  'paper_bgcolor': 'rgba(0,0,0,0)'})

    fig.show()