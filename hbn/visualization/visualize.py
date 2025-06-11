import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

def plotting_style(palette='Paired'):
    plt.style.use('seaborn-v0_8-whitegrid') # ggplot
    params = {'axes.labelsize': 9,
            'axes.titlesize': 15,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.title_fontsize': 10,
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
        hue_order=None,
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
        xticklabels=None,
        marker_color=None,
        bbox_to_anchor=(.2, .4),
        textwrapping_x=True,
        ): 
    if marker_color is not None:
        palette = marker_color
    ax = sns.pointplot(x=x, y=y, hue=hue, data=data, 
                        order=order, ax=ax, join=True, hue_order=hue_order,
                        ci=ci, errwidth=0.5, capsize=0.2, palette=marker_color,
                        )
    for line in ax.lines:
        line.set_linewidth(0.5)
    plt.title(title)
    if not legend:
        ax.get_legend().set_visible(False)
    if hue:
        # set legend to False
        if not legend:
            ax.get_legend().set_visible(False)
        else:
            ax.legend(bbox_to_anchor=bbox_to_anchor, loc=2, borderaxespad=0.0, frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel(y)
    if xlabel is None:
        ax.set_xlabel(x)
    ax.set_ylim(ylim)
    sns.despine()
    ax.text(x_pos, y_pos, subplot, transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    if xticklabels is None:
        ax.xaxis.set_ticklabels(ax.get_xticklabels())
    else:
        ax.xaxis.set_ticklabels(xticklabels)
    # remove x ticks from x axis
    if xticks:
        plt.xticks(rotation=0)
    else:
        plt.xticks([])
    
    if textwrapping_x:
        wrap_xticklabels(ax,10)

    return ax

def barplot( 
            data,
            ax=None,
            y='roc_auc_score',  
            x='development_stage', 
            ylabel='ROC AUC',
            hue=None, 
            hue_order=None,
            order=None, 
            title='', 
            xlabel='',
            subplot=None,
            labelsize=20,
            x_pos=-0.1,
            y_pos=1.1,
            ylim=None,
            legend=True,
            xticks=True,
            xticklabels=None,
            textwrapping_x=True,
            textwrapping_y=False,
            bbox_to_anchor=(.2, .4),
            log_scale=False
            ): 
    import textwrap

    # plot data
    ax = sns.barplot(x=x, y=y, hue=hue, hue_order=hue_order, data=data, order=order, ax=ax, errwidth=0.5, capsize=0.2)
    if log_scale:
        ax.set_yscale('log')
    plt.title(title)
    if hue:
        # set legend to False
        if not legend:
            ax.get_legend().remove()
        else:
            ax.legend(bbox_to_anchor=bbox_to_anchor, loc=2, borderaxespad=0.0, frameon=False)
    if ylabel is None:
        ylabel = y
    if xlabel is None:
        xlabel = x
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.text(x_pos, y_pos, subplot, transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    # xticklabels
    if xticklabels is None:
        xticklabels = ax.get_xticklabels()
    ax.xaxis.set_ticklabels(xticklabels)
    # remove x ticks from x axis
    if not xticks:
        plt.xticks([])
    if textwrapping_x:
        wrap_xticklabels(ax,10)
    if textwrapping_y:
        wrap_yticklabels(ax,15)
    sns.despine()

    plt.tight_layout()

    return ax

def modify_axes(ax1, ax2, ylim_outliers=(40, 80), ylim_data=(0, 10)):

    ax1.set_ylim(ylim_outliers)  # outliers only
    ax2.set_ylim(ylim_data)  # most of the data
    # limit range of y-axis to the data only

    # remove x-axis line's between the two sub-plots
    ax1.spines['bottom'].set_visible(False)  # 1st subplot bottom x-axis
    ax2.spines['top'].set_visible(False)  # 2nd subplot top x-axis

    # 1st x-axis: move ticks from bottom to top
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # no labels
    ax1.tick_params(axis='x', top=False)
    # 2nd x-axis: ticks on the bottom
    ax2.xaxis.tick_bottom()

    # 1st subplot y-axis: remove first tick
    ax1.set_yticks(ax1.get_yticks()[1:])
    # 2nd subplot y-axis: remove the last
    ax2.set_yticks(ax2.get_yticks()[:-1])

    return ax1, ax2

def _draw_cut(ax1, ax2):
    # now draw the cut
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=14,  # "length" of cut-line
        linestyle='none',
        color='k',  # ?
        mec='k',  # ?
        mew=2,  # line thickness
        clip_on=False
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.plot([0, 1], [0.05, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [0, 0.05], transform=ax1.transAxes, **kwargs)

def plot_facetgrid(df, x='Internalizing', col='DX_Reading', hue='Sex', y='data', ylabel=''):

    g = sns.FacetGrid(df, col=col, margin_titles=True, col_wrap=2)
    g.map_dataframe(sns.barplot, x=x, y=y, hue=hue, errwidth=0.5, capsize=0.2,)
    g.set_axis_labels("Assessment", "Score")
    g.add_legend(fontsize=15)

    # get yticklabels
    xticklabels = g.axes[2].get_xticklabels()

    # get xticklabels
    yticklabels = g.axes[2].get_yticklabels()

    axes = g.axes.flatten()
    for i,col in enumerate(df[col].unique()):
        axes[i].set_title(col, fontsize=15)
        axes[i].set_xlabel('')
        axes[i].set_xticklabels(xticklabels, fontsize=10, rotation=45)
        axes[i].set_ylabel(ylabel, fontsize=15)
        axes[i].set_yticklabels(yticklabels, fontsize=10)

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset,ax=None):
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        ) 

def circular_barplot(df, values='value', labels='name', group='group', extra_custom=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Reorder the dataframe
    df_sorted = (
        df \
        .groupby([group])
        .apply(lambda x: x.sort_values([values], ascending = False))
        .reset_index(drop=True)
    )

    VALUES = df_sorted[values].values 
    LABELS = df_sorted[labels].values
    GROUP = df_sorted[group].values

    PAD = 3
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))

    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    GROUPS_SIZE = [len(i[1]) for i in df.groupby(group)]

    offset = 0
    IDXS = []
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

    ax.set_theta_offset(offset)
    # ax.set_ylim(80, 120)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    GROUPS_SIZE = [len(i[1]) for i in df.groupby(group)]
    COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

    ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
        edgecolor="white", linewidth=2
    )

    # add labels for name 
    # padding is the space between the end of the bar and the label
    VALUES_PADDING = []
    for val in VALUES:
        padding = -100
        if val < 100:
            padding = 60
        VALUES_PADDING.append(val + padding)
    add_labels(ANGLES[IDXS], VALUES_PADDING, LABELS, offset, ax=ax)

    # add labels for values
    TEXT = [f'{abs(round(val))}%' for val in VALUES]
    VALUES_PADDING = []
    for val in VALUES:
        padding = 4
        if val < 100:
            padding = 100
        VALUES_PADDING.append( val + padding)
    add_labels(ANGLES[IDXS], VALUES_PADDING, TEXT, offset, ax=ax)

    # Extra customization below here --------------------
    if extra_custom:
        # This iterates over the sizes of the groups adding reference
        # lines and annotations.
        offset = 0 
        num=50
        for group, size in zip(["A", "B", "C", "D"], GROUPS_SIZE):
            # Add line below bars
            x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=num)
            ax.plot(x1, [-5] * num, color="#333333")
            
            # Add text to indicate group
            ax.text(
                np.mean(x1), -20, group, color="#333333", fontsize=14, 
                fontweight="bold", ha="center", va="center"
            )
            
            # Add reference lines at 20, 40, 60, and 80
            x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=num)
            ax.plot(x2, [20] * num, color="#bebebe", lw=0.8)
            ax.plot(x2, [40] * num, color="#bebebe", lw=0.8)
            ax.plot(x2, [60] * num, color="#bebebe", lw=0.8)
            ax.plot(x2, [80] * num, color="#bebebe", lw=0.8)
            
            offset += size + PAD
    
    plt.show()

def wrap_yticklabels(ax, width=12, break_long_words=False):
    import textwrap
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)

    return ax

def wrap_xticklabels(ax, width=12, break_long_words=False):
    import textwrap
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

    return ax

def plot_features_subplot(df, 
                          colors=None,
                          x='feature_importances', 
                          y='key', 
                          hue=None,
                          top_features=10,
                          ax=None,
                          topic_labels=False,
                          x_pos=-0.1,
                          y_pos=1.1,
                          labelsize=20,
                          title='',
                          subplot='A'
                          ):
    
    if hue is not None:
        df = df.groupby(hue)\
                .apply(lambda x: x[['feature_importances', 'feature_names', hue]] \
                    .sort_values(by='feature_importances', ascending=False) \
                        .head(top_features)).reset_index(drop=True)
    else:
        df = df.sort_values(by='feature_importances', ascending=False).head(top_features)

    # plot data
    ax = sns.barplot(x=x, y=y, data=df, hue=hue, ax=ax)

    if topic_labels:
        for ytick in ax.get_yticklabels():
            key = ytick.get_text()
            topic = df[df[y]==key]['custom_labels'].unique()[0]
            ytick.set_color(colors[topic])

    ax.set_title(title)
    ax.set_ylabel(f'Top {top_features} features')
    ax.set_xlabel('Feature Importance') 
    ax.text(x_pos, y_pos, subplot, transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    wrap_yticklabels(ax,12)

    plt.tight_layout()
    sns.despine()

def rgb_to_hex(rgb_list):
  """
  Converts a list of RGB color tuples to a list of corresponding hexadecimal strings.

  Args:
    rgb_list: A list of tuples, where each tuple represents an RGB color 
              with three integers (red, green, blue) ranging from 0 to 255.

  Returns:
    A list of strings, where each string is the hexadecimal color code 
    (e.g., "#FFFFFF").
  """
  hex_colors = []
  for r, g, b in rgb_list:
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    hex_colors.append(hex_color)
  return hex_colors


def plot_stacked(
        df, 
        ylabel='Selected Features (%)',
        ax=None,
        subplot='',
        colors=None,
        xlabel='',
        labelsize=20,
        x_pos=-0.1,
        y_pos=1.2,
        ylim=[0.4,1],
        legend=True,
        xticks=True,
        xticklabels=None,
        textwrapping_x=True,
        textwrapping_y=False,
        bbox_to_anchor=(.2, .4)
        ): 
    """plot % of features for each topic for selected model and overall model (features input to model)"""
    if ax is None:
        ax = plt.subplot(111)
    
    # define plot
    df.plot(kind='bar', legend=True, stacked=True, width=.9, ax=ax)
    if colors is not None:
        df.plot(kind='bar', legend=True, stacked=True, width=.9, colors=colors, ax=ax)

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
    # xticklabels
    if xticklabels is None:
        xticklabels = ax.get_xticklabels()
    ax.xaxis.set_ticklabels(xticklabels)
    # remove x ticks from x axis
    if not xticks:
        plt.xticks([])
    if textwrapping_x:
        wrap_xticklabels(ax,10)
    if textwrapping_y:
        wrap_yticklabels(ax,10)

    plt.tight_layout()
    sns.despine()

    return ax


def calculate_slopes(
    df,
    compare_group={'DX_Subtype_Name': 'ADHD-Combined Type', 'sex': ['male', 'female']}, 
    overall_group={'DX_Cat_Name_New': 'ADHD', 'sex': ['male', 'female']},
    groupby=['age_round']
    ):
    from hbn.visualization import raw_scores
    from hbn.features.build_features import _index_dataframe_by_columns_values

    def _filter_data(df, dict):
        cols = []; vals = []
        for k, v in dict.items():
            cols.append(k)
            if not isinstance(v, list):
                v = [v]
            vals.append(v)

        df_out = _index_dataframe_by_columns_values(dataframe=df, columns=cols, values_list=vals)
        df_out = df_out.groupby(['Identifiers']).first().reset_index(drop=True)
        
        return df_out

    # filter data
    df_compare = _filter_data(df, compare_group)
    df_overall = _filter_data(df, overall_group)

    # calculate female percentage
    df_perc = raw_scores.calculate_sex_percentage(df_compare, df_overall, groupby=groupby)

    return df_perc


def plot_slopes(
        df, 
        ax=None,
        x='age_round', 
        y='perc', 
        xlabel='Age', 
        ylabel=None, 
        hue=None, 
        title='',
        subplot=None,
        labelsize=20,
        x_pos=-0.1,
        y_pos=1.2,
        legend=True,
        xticks=True,
        xticklabels=None,
        bbox_to_anchor=(.2, .4),
        col=None, 
        height=4, 
        aspect=1, 
        marker='o',
        fit_reg=True,
        ylim=[0,100],
        linestyle='-',
        textwrapping_x=True,
        textwrapping_y=False
        ): 

    scatter = True
    if marker is None:
        scatter=False
    
    # plot data
    if ax is None:
        fg = sns.lmplot(x=x, y=y, data=df, hue=hue, col=col, height=height, scatter=scatter, scatter_kws={'s': 90}, aspect=aspect)
        ax = fg.axes[0, 0] 
    else:
        if hue is not None:
            hues = df[hue].unique()
            colors = sns.color_palette('Paired', len(hues))
            for (val, color) in zip(hues, colors):
                ax = sns.regplot(x=x, y=y, data=df[df[hue]==val], color=color, 
                                 marker=marker, scatter=scatter, scatter_kws={'s': 90}, ax=ax,
                                 line_kws={'linestyle': linestyle}, label=val, fit_reg=True)    # label=val, 
        else:
            ax = sns.regplot(x=x, y=y, data=df, ax=ax)
    if fit_reg:
        # fit regression line
        sns.regplot(x=x, y=y, data=df, fit_reg=True, scatter=False, 
                color='lightgray', line_kws={'color': 'lightgray', 'linestyle': '--','alpha': 0.6}, ci=95, ax=ax)
    plt.title(title)
    if hue:
        # set legend to False
        if not legend:
            ax.legend().remove()
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
    # xticklabels
    if xticklabels is None:
        xticklabels = ax.get_xticklabels()
    ax.xaxis.set_ticklabels(xticklabels)
    # remove x ticks from x axis
    if not xticks:
        plt.xticks([])
    if textwrapping_x:
        wrap_xticklabels(ax,10)
    if textwrapping_y:
        wrap_yticklabels(ax,15)
    sns.despine()
        
    # from hbn.visualization import stats
    # stats.compare_slopes(df_perc[df_perc['sex']=='female'], df_perc[df_perc['sex']=='male'], alpha=0.05, x='age_round', y='perc')

    return ax

def scatterplot(df, 
        x, 
        y, 
        ax=None,
        hue=None, 
        hue_order=None,
        style=None,
        xlabel=None, 
        ylabel=None, 
        subplot=None,
        labelsize=20,
        x_pos=-0.1,
        y_pos=1.1,
        ylim=None,
        xlim=None,
        legend=True,
        xticks=True,
        fit_reg=True,
        bbox_to_anchor=(.2, .4),
        ):
    
    ax = sns.scatterplot(x=x, y=y, hue=hue, style=style, data=df, hue_order=hue_order, ax=ax)

    if fit_reg:
        # fit regression line
        sns.regplot(x=x, y=y, data=df, fit_reg=True, scatter=False, 
                color='lightgray', line_kws={'color': 'lightgray', 'linestyle': '--','alpha': 0.6}, ci=95, ax=ax)

    # set legend to False
    if not legend:
        ax.get_legend().remove()
    else:
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=2, borderaxespad=0.0)
    if ylabel is None:
        ylabel = y
    if xlabel is None:
        xlabel = x
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.text(x_pos, y_pos, subplot, transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    # remove x ticks from x axis
    if not xticks:
        plt.xticks([])

    plt.tight_layout()
    sns.despine()

    