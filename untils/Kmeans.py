import numpy as np
from sklearn.cluster import KMeans
from encoder.Encoder import encoder
from sklearn.metrics.pairwise import cosine_similarity

def get_most_different_chains_by_clustering(chains,answers, k):
    if not chains or k <= 0 or k > len(chains):
        return []

    chain_matrix = encoder(chains)
    if chain_matrix.ndim == 1:
        chain_matrix = chain_matrix.reshape(-1, 1)

    n = len(chains)
    if n == 1:
        return [chains[0]]

    kmeans = KMeans(n_clusters=min(k, n), random_state=42,n_init=10)
    kmeans.fit(chain_matrix)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_


    selected_indices = []
    for cluster_idx in range(min(k, n)):

        cluster_indices = np.where(labels == cluster_idx)[0]
        if len(cluster_indices) == 0:
            continue


        cluster_center = centers[cluster_idx].reshape(1, -1)

        cluster_vectors = chain_matrix[cluster_indices]


        similarities = cosine_similarity(cluster_vectors, cluster_center)
        distances = 1 - similarities.flatten()

        closest_idx = cluster_indices[np.argmin(distances)]
        selected_indices.append(closest_idx)
        cot=[chains[idx] for idx in selected_indices]
        Answers=[answers[idx] for idx in selected_indices]

    return cot,Answers

