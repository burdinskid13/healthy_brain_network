
def compare_answers(feature_specs, age='all', sex='all', diagnoses=None, split='all'):
    """compare answers across all combinations of `feature_specs`
    
    Args:
        feature_specs (list of str): fullpaths to feature specs 
        age (list of int or str): (optional): default is 'all'. other options are any numbers between 6 - 21
        sex (str): (optinal): default is 'all'. other options: 'female', 'male' 
        diagnoses (list of str): (optional) list of diagnoses from `model_specs`. for example `diagnoses = ['ADHD-Combined Type']`
        split (str): (optional). default is 'all'. other options: 'train', 'test'
    Returns:
        `df` (pandas dataframe): 
    """
    import itertools
    import dcor
    import pandas as pd
    from pathlib import Path

    corr_all = []; measures_all = []; features_all = []
    for combo in itertools.combinations(feature_specs, 2):

        # get features
        features = [Path(c).name for c in combo]

        # get answers
        df_A = get_answers(
            feature_specs=list(combo), 
            age=age,
            sex=sex,
            diagnoses=diagnoses, 
            split=split
            )

        x = df_A[0]; y = df_A[1]

        # get measure
        measures = []
        for comb in combo:
            measure = Path(comb).stem.split("-")[-2]
            measures.append(measure)
        measures = '/'.join(measures)

        # calculate distance correlation
        corr_A = dcor.distance_correlation(x, y)
        corr_all.append(corr_A)
        measures_all.append(measures)
        features_all.append(features)

    df = pd.DataFrame({
            'features': features_all,
            'measures': measures_all,
            'distances': corr_all
            })
    
    return df


def get_answers(feature_specs, age='all', sex='all', diagnoses=None, split='all'):
    """get answers from `feature_specs` returned as a list of dataframes
    
    Args:
        feature_specs (list of str): fullpaths to feature specs
        age (list of int or str): (optional): default is 'all'. other options are any numbers between 6 - 21
        sex (str): (optional): default is 'all'. other options: 'female', 'male' 
        diagnoses (list of str): (optional) list of diagnoses from `model_specs`. for example `diagnoses = ['ADHD-Combined Type']`
    Returns:
        df_list (list of pd dataframes): list of dataframes
    """
    import os
    from hbn import io
    from functools import reduce
    from hbn.data import make_dataset
    from hbn.features import feature_selection
    from hbn.constants import Defaults
    
    if diagnoses is None:
        diagnoses = ['No_Diagnosis_Given']
        
    participants = make_dataset.get_participants(
                                split=split, 
                                disorders=diagnoses,
                                age=age,
                                sex=sex
                                )
    # loop over feature specs 
    df_list = []; same_part = []
    for feature_spec in feature_specs:

        # load feature spec
        spec_info = io.read_json(os.path.join(Defaults.FEATURE_DIR, feature_spec))
        spec_info['preprocessing'] = None # not doing preprocessing as defined in the feature spec file

        # get data
        df = feature_selection.phenotype_features(target_spec=None,
                                feature_spec=spec_info,
                                participants=participants,
                                preprocess=True,
                                drop_identifiers=False,
                                oversample=False
                                ).reset_index(drop=True)
        df = df.drop_duplicates(subset=['Identifiers'])
        df_list.append(df)
        same_part.append(df['Identifiers'])

    part_intersect = list(reduce(set.intersection, map(set, same_part)))

    # index dataframes for `part_intersect`
    df_out = []
    for df in df_list:
        df_part = df[df['Identifiers'].isin(part_intersect)]
        df_part = df_part.drop(columns=['Identifiers'])
        df_out.append(df_part)
    
    return df_out


def get_questions(feature_specs):
    """get questions for `feature_specs`, reads `datadic` key and loads dictionary
    
    Args:
        feature_specs (list of str): fullpaths to feature specs
    Returns:
        `df_list` (list of pandas dataframes)
    """
    import os
    from hbn import io
    import pandas as pd
    
    # loop over features
    df_list = []
    for feature_spec in feature_specs:
        spec_info = io.read_json(os.path.join(Defaults.FEATURE_DIR, feature_spec))
        
        # get abbreviations
        abbrev = spec_info['datadic']
        
        # load in excel file of data dictionary
        df_Q = pd.read_excel(os.path.join(Defaults.PHENO_DIR, 'Release9_DataDic',  f'{abbrev}.xlsx'), header=1)
        df_Q.columns = [col.strip(' ') for col in df_Q]
        df_list.append(df_Q)
            
    return df_list
            

def sentence_similarity(sentences):
    """calculate sentence similarity across all sentence combinations
    
    Args:
        sentences (list): list of sentences
    """
    from sentence_transformers import SentenceTransformer, util
    
    # get model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    # calculate sentence embeddings
    sentence_embeddings = model.encode(sentences)
    
    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings)
    
    #Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    
    return cosine_scores, pairs


def answer_similarity(dataframe, pairs):
    """calculate correlations across answers for all `pairs` of sentences
    
    Args:
        dataframe (pd dataframe):
        pairs (list of dict): output from `sentence_similarity`
    Returns:
        pairs_A (list)
    """
    
    # calculate correlations across answers
    df_corr = dataframe.corr()
    
    pairs_A = []
    for idx in range(len(pairs)):
        r = pairs[idx]['index'][0]
        c = pairs[idx]['index'][1]
        rcorr = df_corr.iloc[r,c]
        pairs_A.append(rcorr)
    
    return pairs_A


def distance_correlation_permutation(x, y, n_tests=100, n_samples=1000, dim=30, significance=0.1, n_obs_list=[25, 30, 35, 50, 70, 100]):
    """calculate permuaton of distance correlation between two np arrays (x and y), # of rows must match across x and y
        # of columns can be different across x and y
    
    Args:
        x (np array): rows x columns 
        y (np array): rows x columns
    Returns:
        `table` table of results
    """
    import scipy
    import dcor
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt

    table = pd.DataFrame()
    table["n_obs"] = n_obs_list

    dist_results = []
    for n_obs in n_obs_list:
        n_errors = 0
        statistics = []
        for _ in range(n_tests):
            x = np.random.normal(0, 1, size=(n_samples, dim))
            y = np.random.normal(0, 1, size=(n_samples, dim))

            test_result = dcor.independence.distance_correlation_t_test(x, y)
            statistics.append(test_result.statistic)
            if test_result.pvalue < significance:
                n_errors += 1

        error_prob = n_errors / n_tests
        dist_results.append(error_prob)

    table["Type I error"] = dist_results

    # Plot the last distribution of the statistic
    df = len(x) * (len(x) - 3) / 2

    plt.hist(statistics, bins=12, density=True)

    distribution = scipy.stats.t(df=df)
    u = np.linspace(distribution.ppf(0.01), distribution.ppf(0.99), 100)
    plt.plot(u, distribution.pdf(u))
    plt.show()

    table


def distance_covariance_permutation(x,y):
    import time
    import dcor
    from matplotlib import pyplot as plt
    
    random_state = 42
    num_resamples_list = [10, 50, 100, 200, 500]

    pvalues = []
    times = []

    for num_resamples in num_resamples_list:

        start_time = time.monotonic()
        test_result = dcor.independence.distance_covariance_test(
            x,
            y,
            num_resamples=num_resamples,
            random_state=random_state,
        )
        end_time = time.monotonic()

        pvalues.append(test_result.pvalue)
        times.append(end_time - start_time)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(num_resamples_list, pvalues)
    axes[1].plot(num_resamples_list, times, color="C1")
    axes[1].set_xticks(num_resamples_list)
    axes[1].set_xlabel("number of permutations")
    axes[0].set_ylabel("p-value")
    axes[1].set_ylabel("time (in seconds)")
    plt.show()


def plot_heatmap(array, labels=None):
    """plot heatmap of array with labels (optional)
    
    Args:
        array (np array):
        labels (list of str or None): optional
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    ax = sns.heatmap(array)
    
    if labels is not None:
        ax.set_xticklabels(
            labels,
            rotation=45,
            horizontalalignment='right'
            )
        ax.set_yticklabels(
            labels,
            rotation=360,
            horizontalalignment='right'
        )

    plt.show()
    

def dendrogram_plot(dataframe, method='ward', metric='euclidean', labels=None, orientation='top', color_leaves=True, ax=None):
    """ Plots dendrogram plot.
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            method (str): 'ward' # there are other options given by linkage
            metric (str): 'euclidean' # there are other options given by linkage
            ax (bool): figure axes. Default is None
            color_leaves (bool): whether or not leaf nodes should be colored
            orientation (str): how the dendrogram should be oriented. Default is 'top'. Other options are 'right', 'left'
            reorder (bool): reorders the expression matrix before being input to clustering algo
    """
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
    from matplotlib import pyplot as plt
    
    if ax is None:
        plt.figure(num=1, figsize=[25,8])

    if color_leaves:
        set_link_color_palette(['b', 'r', 'y', 'm'])
    else:
        set_link_color_palette(['k', 'k', 'k', 'k'])
        
    if labels is not None:
        labels = dataframe.index.to_list()

    R = dendrogram(
        Z=linkage(dataframe, method, metric),
        orientation=orientation,
        get_leaves=True,
        color_threshold=35.0,
        labels=labels,
        distance_sort='ascending',
        above_threshold_color='black', 
        ax=ax, 
        )

    # plt.title("Hierarchical Clustering Dendrogram", fontsize=20)
    plt.xlabel('')
    plt.ylabel(f"{metric.capitalize()} Distance")

    return R


def plot_top_sentences(pairs, sentences, A_similarity=None, percent=10, title=None):
    """plot most similar sentences in a tabular format
    
    Args:
        pairs (list of dict): output from `sentence similarity`
        sentences (list of str): list of sentences
        A_similarity (list of str or None): (optional) output from `answer_similarity`
        percent (int): percentage of top sentences to print
    """
    from tabulate import tabulate
    
    pairs_all = pairs
    header = ["Sentence 1", "Sentence 2", "Question"]
    
    if A_similarity is not None:
        # add answer similarity score to `pairs` dictionary
        pairs_all = []
        header = ["Sentence 1", "Sentence 2", "Question", "Answer"]
        for r, score in zip(pairs, A_similarity):
            r.update({'score_A': score})
            pairs_all.append(r)

    #Output the pairs with their score
    top_sentences = []
    num_sentences = int(len(sentences)*percent/100)
    for pair in pairs_all[0:num_sentences]:
        i, j = pair['index']
        if A_similarity is not None:
            row = [sentences[i], sentences[j], round(pair['score'].tolist(),2), round(pair['score_A'],2)]
        else:
            row = [sentences[i], sentences[j], round(pair['score'].tolist(),2)]
        top_sentences.append(row)
         
    top_sentences.insert(0, header)
    head='firstrow'
    if title is not None:
        head = [title, '','','']
    print(tabulate(top_sentences, headers=head, tablefmt="grid"))

    
def num_resamples_from_obs(n_obs):
    return 200 + 5000 // n_obs


def multivariate_normal(n_obs, dim=5):
    import numpy as np

    return np.random.normal(size=(n_obs, dim),)


def t_dist_generator(df):
    import numpy as np
    def t_dist(n_obs, dim=5):
        return np.random.standard_t(
            df=df,
            size=(n_obs, dim),
        )

    return t_dist


def monte_carlo_test(x,y, n_tests=100, n_obs_list=[25, 30, 35, 50, 70, 100], significance=0.1):
    import dcor
    import pandas as pd
    
    num_resamples_list = [10, 50, 100, 200, 500]
    
    distributions = {
        "Multivariate normal": multivariate_normal,
        "t(1)": t_dist_generator(1),
        "t(2)": t_dist_generator(2),
        "t(3)": t_dist_generator(3),
    }
    table = pd.DataFrame()
    table["n_obs"] = n_obs_list
    table["num_resamples"] = num_resamples_list

    for dist_name, dist in distributions.items():
        dist_results = []
        for n_obs, num_resamples in zip(n_obs_list, num_resamples_list):
            n_errors = 0
            for _ in range(n_tests):
                x = dist(n_obs)
                y = dist(n_obs)

                test_result = dcor.independence.distance_covariance_test(
                    x,
                    y,
                    num_resamples=num_resamples,
                    random_state=42,
                )

                if test_result.pvalue < significance:
                    n_errors += 1

            error_prob = n_errors / n_tests
            dist_results.append(error_prob)

        table[dist_name] = dist_results

    table

