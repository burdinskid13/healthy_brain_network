import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

def plotting_style():
    plt.style.use('seaborn-poster') # ggplot
    params = {'axes.labelsize': 20,
            'axes.titlesize': 25,
            'legend.fontsize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.title_fontsize': 20,
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


def violinplot(x, 
               data, 
               y='f1_score', 
               ylim=[0.3, 1], 
               hue=None, 
               fig_title='', 
               legend_title='',
               xlabel='', 
               ylabel='', 
               x_order=None,
               figsize=(3,3)
               ):
    split = False
    if hue is not None:
        split = True
    plt.figure(figsize=figsize)
    ax = sns.violinplot(data=data, x=x, y=y, hue=hue, split=split, order=x_order);
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(y)
    ax.set_xlabel(xlabel)
    ax.set_title(fig_title)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(ylim)
    if hue is not None:
        plt.legend(loc='best', fontsize=10, bbox_to_anchor=(1.3, 1.05), title=legend_title);
    sns.despine(bottom=False, left=False)
    plt.tight_layout()

    return ax

def lineplot(x, 
             data, 
             y='f1_score', 
             ylim=[0.3, 1], 
             hue=None, 
             style=None,  
             fig_title='', 
             legend_title='Model', 
             xlabel='', 
             ylabel='', 
             palette=None, 
             figsize=(3,3)
             ):
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, style=style, err_style='bars', palette=palette);
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(y)
    ax.set_xlabel(xlabel)
    ax.set_title(fig_title)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(ylim)
    if hue is not None:
        plt.legend(loc='best', fontsize=10, bbox_to_anchor=(1.5, 1.05), title=legend_title);
    plt.tight_layout()

    sns.despine(bottom=False, left=False)
    return ax

def barplot(data, 
            y='feature_names', 
            x='feature_importances', 
            hue=None, 
            title='', 
            xlabel='Feature Importance', 
            ylabel=None, 
            top_features=20, 
            figsize=(3,3)
            ):
    plt.figure(figsize=figsize)
    if y=='feature_names':
        data['feature_names'] = data['feature_names'].str.replace('numeric__', '')
        ax = sns.barplot(y=y, x=x, hue=hue,data=data.head(top_features))
    else:
        ax = sns.barplot(y=y, x=x, hue=hue, data=data)

    if ylabel is None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(y)
    if xlabel is None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(x)
    ax.set_title(title)
    sns.despine(bottom=False, left=False)
    plt.tight_layout()
    return ax

def pointplot(
            data, 
            y='feature_names', 
            x='feature_importances', 
            hue=None,
            fig_title='', 
            legend_title='Model', 
            ylim=[0.3, 1], 
            xlabel='ROC AUC', 
            ylabel=None, 
            markers="o",
            linestyles="--",
            join_lines=True,
            scale=0.5,
            dodge=.4,
            alpha=.2,
            figsize=(3,3)
            ):
    plt.figure(figsize=figsize)
    ax = sns.stripplot(
        data=data, x=x, y=y, hue=hue,
        dodge=True, alpha=alpha, legend=False,
        )
    sns.pointplot(
        data=data, x=x, y=y, hue=hue,
        join=join_lines, dodge=dodge, errorbar=None,
        markers=markers, scale=scale,
        linestyles=linestyles, 
    )

    # customize hue
    if hue is not None:
        plt.legend(loc='best', fontsize=10, bbox_to_anchor=(1.1, 1.05), title=legend_title);

    # Customize the plot further (optional)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(fig_title)
    plt.xticks(ha='right', rotation=45)
    sns.despine(bottom=False, left=False)
    plt.tight_layout()
    
    return ax

    