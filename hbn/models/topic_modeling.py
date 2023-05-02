## code taken from this tutorial: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

def get_embeddings(
    data, 
    transformer='distilbert-base-nli-mean-tokens'):
    from sentence_transformers import SentenceTransformer	

    model = SentenceTransformer(transformer)
    embeddings = model.encode(data, show_progress_bar=True)

    return embeddings


def dimensionality_reduction(
    embeddings, 
    n_neighbors=15, 
    n_components=5,
    metric='cosine',
    min_dist=0.1
    ):
    import umap
    umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
                            n_components=n_components, 
                            metric=metric,
                            min_dist=min_dist).fit_transform(embeddings)

    return umap_embeddings


def clustering(
    umap_embeddings,
    min_cluster_size=15,
    metric='euclidean',
    cluster_selection_method='eom'
    ):
    import hdbscan
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            metric=metric,                      
                            cluster_selection_method=cluster_selection_method).fit(umap_embeddings)

    return cluster

def visualize_clusters(
    embeddings,
    cluster,
    ):
    import pandas as pd
    import matplotlib.pyplot as plt
    import umap

    # Prepare data
    umap_data = dimensionality_reduction(embeddings, 
                                        n_neighbors=15, 
                                        n_components=2, 
                                        min_dist=0.0, 
                                        metric='cosine'
                                        )

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.show()


def tf_idf(
        data,
        cluster
        ):
    import pandas as pd

    docs_df = pd.DataFrame(data, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    return docs_per_topic, docs_df

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def topic_reduction(
    data, 
    docs_df
    ):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    for i in range(20):
        # Calculate cosine similarity
        similarities = cosine_similarity(tf_idf.T)
        np.fill_diagonal(similarities, 0)

        # Extract label to merge into and from where
        topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
        topic_to_merge = topic_sizes.iloc[-1].Topic
        topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

        # Adjust topics
        docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
        old_topics = docs_df.sort_values("Topic").Topic.unique()
        map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
        docs_df.Topic = docs_df.Topic.map(map_topics)
        docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

        # Calculate new topic words
        m = len(data)
        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

    return docs_df, top_n_words
