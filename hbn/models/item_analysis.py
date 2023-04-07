
def load_data(assessments=['Child', 'Parent'], data_type='preprocessed'):
    import pandas as pd
    import os
    """load data for answers for `assessment` and `data_type`
    Args:   
        data_dir (str): full path to directory where data (csv files) are saved
        assessment (str): Default is 'Child'. other options: 'Parent', 'Teacher'
        data_type (str): Default is 'preprocessed'. Other option is 'raw'
    Returns:
        df_data (pd dataframe), df_dict (pd dataframe): containing questionnaire answers and dictionary keys respectively
    """
    from hbn.constants import Defaults

    df_diagnosis_all = pd.DataFrame()
    df_dict_all = pd.DataFrame()
    df_data_all = pd.DataFrame()
    for assessment in assessments:
        # load csv
        df_data = pd.read_csv(os.path.join(Defaults.SUBTYPE_DIR, f'{assessment}-features-{data_type}.csv'))
        df_data.columns = df_data.columns.str.replace('numeric__', '')

        # return dictionary
        df_dict = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'item-names-cleaned.csv'))
        df_dict = df_dict[df_dict['assessment']==f'{assessment} Measures']

        # return clinical diagnosis + demographics
        df_diagnosis = pd.read_csv(os.path.join(Defaults.PHENO_DIR, 'Clinical_Measures', 'Clinical_Diagnosis_Demographics.csv'))
        df_diagnosis = df_diagnosis.rename(columns={'DX_01': 'Diagnosis', 
                                'DX_01_Cat_new': 'Category', 
                                'PreInt_Demos_Fam,Child_Race_cat': 'Race',
                                'PreInt_Demos_Fam,Child_Ethnicity_cat': 'Ethnicity'
                                })

        df_diagnosis_all = pd.concat([df_diagnosis_all, df_diagnosis])
        df_data_all = pd.concat([df_data_all, df_data])
        df_dict_all = pd.concat([df_dict_all, df_dict])

    return df_data_all, df_dict_all, df_diagnosis_all
            
            
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