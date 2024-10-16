import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

def plotting_style(palette='Paired'):
    plt.style.use('seaborn-poster') # ggplot
    params = {'axes.labelsize': 10,
            'axes.titlesize': 15,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.title_fontsize': 20,
            # 'figure.figsize': (10,5),
            'font.weight': 'regular',
            # 'font.size': 'regular',
            'font.family': 'sans-serif',
            'lines.markersize': 10,
            'font.serif': 'Helvetica Neue',
            'lines.linewidth': 2,
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False}
    plt.rcParams.update(params)
    sns.set_context(rc={'lines.markeredgewidth': 0.1})
    sns.set_palette(palette)
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
    # plt.axhline(y=0.5, color='black', linestyle='--')
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


def pointplot(
        ax, 
        data, 
        y='roc_auc_score',
        x='development_stage', 
        hue=None, 
        order=None, 
        title='', 
        ylim=[0.4,1],
        ylabel='ROC AUC',
        xlabel='',
        subplot=None,
        labelsize=20,
        x_pos=-0.1,
        y_pos=1.1,
        ci=95,
        legend=True,
        xticks=True,
        marker_color=None,
        bbox_to_anchor=(.2, .4)
        ): 
    if marker_color is not None:
        palette = marker_color
    ax = sns.pointplot(x=x, y=y, hue=hue, data=data, 
                        order=order, ax=ax, join=legend, 
                        ci=ci, errwidth=0.5, capsize=0.2, palette=marker_color,
                        )
    plt.title(title)
    if hue:
        # set legend to False
        if not legend:
            ax.get_legend().set_visible(False)
        else:
            ax.legend(bbox_to_anchor=bbox_to_anchor, loc=2, borderaxespad=0.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel(y)
    if xlabel is None:
        ax.set_xlabel(x)
    ax.set_ylim(ylim)
    sns.despine()
    ax.text(x_pos, y_pos, subplot, transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    # remove x ticks from x axis
    if xticks:
        plt.xticks(rotation=45, ha='right')
    else:
        ax.set_xticklabels([])

    return ax

def barplot(ax, 
            data,
            y='roc_auc_score',  
            x='development_stage', 
            hue=None, 
            order=None, 
            title='', 
            ylabel='ROC AUC',
            xlabel='',
            subplot=None,
            labelsize=20,
            x_pos=-0.1,
            y_pos=1.1,
            ylim=[0.4,1],
            legend=True,
            xticks=True,
            bbox_to_anchor=(.2, .4)
            ): 
    # plot data
    ax = sns.barplot(x=x, y=y, hue=hue, data=data, order=order, ax=ax, errwidth=0.5, capsize=0.2)
    plt.title(title)
    if hue:
        # set legend to False
        if not legend:
            ax.get_legend().remove()
        else:
            ax.legend(bbox_to_anchor=bbox_to_anchor, loc=2, borderaxespad=0.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel(y)
    if xlabel is None:
        ax.set_xlabel(x)
    ax.set_ylim(ylim)
    ax.text(x_pos, y_pos, subplot, transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    # remove x ticks from x axis
    if xticks:
        plt.xticks(rotation=45, ha='right')
    else:
        ax.set_xticklabels([])
    sns.despine()

    return ax

def plot_facetgrid(df, data='Internalizing'):

    g = sns.FacetGrid(df, col="DX_Reading", margin_titles=True, col_wrap=2)
    g.map_dataframe(sns.barplot,x=data, y=f'{data}_standarized', hue='Sex', errwidth=0.5, capsize=0.2,)
    g.set_axis_labels("Assessment", "Score")
    g.add_legend(fontsize=15)

    # get yticklabels
    xticklabels = g.axes[2].get_xticklabels()

    # get xticklabels
    yticklabels = g.axes[2].get_yticklabels()

    axes = g.axes.flatten()
    for i,col in enumerate(df['DX_Reading'].unique()):
        axes[i].set_title(col, fontsize=15)
        axes[i].set_xlabel('')
        axes[i].set_xticklabels(xticklabels, fontsize=10, rotation=45)
        axes[i].set_ylabel(f'{data} Score', fontsize=15)
        axes[i].set_yticklabels(yticklabels, fontsize=10)

    