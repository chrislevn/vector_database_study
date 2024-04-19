import numpy as np
from sklearn.cluster import MiniBatchKMeans
import hnswlib

class Flat:
    def __init__(self, sentence_embeddings):
        """
        Flat Index
        
        Args:
            sentence_embeddings (np.array): Array of sentence embeddings
            
        """
        self.sentence_embeddings = sentence_embeddings

    def search(self, query_embedding, top_k=1):
        """
        Search for the most similar sentences to a query embedding
        
        Args:
            query_embedding (np.array): Query embedding
            top_k (int): Number of similar sentences to return
        
        Returns:
            list: List of indices of similar sentences
        """
        similarities = np.dot(self.sentence_embeddings, query_embedding) / (np.linalg.norm(self.sentence_embeddings, axis=1) * np.linalg.norm(query_embedding))
        sorted_indices = np.argsort(-similarities)
        return sorted_indices[:top_k]
    
# create ivf index from scratch based on sentence_embeddings and cosine similarity
class IVF:
    def __init__(self, sentence_embeddings, n_clusters=100):
        """
        Inverted File Index
        
        Args:
            sentence_embeddings (np.array): Array of sentence embeddings
            n_clusters (int): Number of clusters to use for k-means clustering
            
        """
        # Cluster the sentence embeddings using k-means
        kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        kmeans.fit(sentence_embeddings)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.cluster_labels_ = kmeans.labels_
        self.sentence_embeddings = sentence_embeddings

        # Create an inverted index
        self.inverted_index = {}
        for i, label in enumerate(self.cluster_labels_):
            if label not in self.inverted_index:
                self.inverted_index[label] = []
            self.inverted_index[label].append(i)

    def search(self, query_embedding, top_k=1):
        """
        Search for the most similar sentences to a query embedding
        
        Args:
            query_embedding (np.array): Query embedding
            top_k (int): Number of similar sentences to return
        
        Returns:
            list: List of indices of similar sentences
        """
        # Find the cluster closest to the query embedding
        distances = np.linalg.norm(self.cluster_centers_ - query_embedding, axis=1)
        closest_cluster_index = np.argmin(distances)

        # Search for the top k results within the closest cluster
        top_k_indices = []
        for i in self.inverted_index[closest_cluster_index]:
            cosine_similarity_scores = np.dot(query_embedding, self.sentence_embeddings[i])/(np.linalg.norm(query_embedding)*np.linalg.norm(self.sentence_embeddings[i]))
            top_k_indices.append((i, cosine_similarity_scores))
        top_k_indices.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [index for index, _ in top_k_indices[:top_k]]

        return top_k_indices


class HNSW: 
    def __init__(self, sentence_embeddings, sentences, M=16, ef_construction=200, metric='cosine'): 
        """
        HNSW Index
        
        Args:
            sentence_embeddings (np.array): Array of sentence embeddings
            sentences (list): List of sentences
            M (int): Number of neighbors in the graph
            ef_construction (int): Number of neighbors to consider during construction
            metric (str): Similarity metric to use
        """
        # Define the HNSW parameters
        dim = sentence_embeddings.shape[1]  # Dimensionality of the vectors
        ids = np.arange(sentence_embeddings.shape[0])

        # Create the HNSW index
        index = hnswlib.Index(space=metric, dim=dim)
        index.init_index(max_elements = len(sentences), ef_construction = ef_construction, M = M)

        # Add the sentence embeddings to the index
        index.add_items(sentence_embeddings, ids)

        # Return the index

        self.index = index
        self.sentences = sentences

    def search(self, query_embedding, top_k=1):
        # Perform a search using a query embedding
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        top_k_indices = labels[0]
        return top_k_indices

class ProductQuantization:
    def __init__(self, sentence_embeddings, n_subvectors=8):
        """
        Product Quantization Index
        
        Args:
            sentence_embeddings (np.array): Array of sentence embeddings
            n_subvectors (int): Number of subvectors to divide each embedding into
        """
        # Subspace dimensionality
        self.d = sentence_embeddings.shape[1] // n_subvectors
        # Number of subvectors
        self.m = n_subvectors

        # Generate random projection matrix
        self.R = np.random.randn(self.d * self.m, self.d)
        self.R /= np.linalg.norm(self.R, axis=0)

        # Project sentence embeddings onto subspaces
        self.subspace_embeddings = np.dot(sentence_embeddings, self.R)

        # Quantize subspace embeddings
        self.quantized_subspace_embeddings = np.round(self.subspace_embeddings / self.d)

    def search(self, query_embedding, top_k=1):
        """
        Search for the most similar sentences to a query embedding

        Args:
            query_embedding (np.array): Query embedding
            top_k (int): Number of similar sentences to return

        Returns:
            list: List of indices of similar sentences
        """ 
        # Project query embedding onto subspaces
        query_subspace_embeddings = np.dot(query_embedding, self.R)

        # Quantize query subspace embeddings
        quantized_query_subspace_embeddings = np.round(query_subspace_embeddings / self.d)

        # Find nearest neighbors in each subspace
        nearest_neighbors = []
        for i in range(self.m):
            distances = np.linalg.norm(self.quantized_subspace_embeddings[:, i] - quantized_query_subspace_embeddings[i], ord=2)
            nearest_neighbors.append(np.argsort(distances)[:top_k])

        # Combine nearest neighbors from each subspace
        combined_nearest_neighbors = set()
        for neighbors in nearest_neighbors:
            combined_nearest_neighbors.update(neighbors)

        return list(combined_nearest_neighbors)
    
class ScalarQuantization:
    def __init__(self, sentence_embeddings, n_clusters=100):
        """ 
        Scalar Quantization Index
        
        Args:
            sentence_embeddings (np.array): Array of sentence embeddings
            n_clusters (int): Number of clusters to use for k-means clustering
        """
        # Cluster the sentence embeddings using k-means
        kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        kmeans.fit(sentence_embeddings)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.cluster_labels_ = kmeans.labels_
        self.sentence_embeddings = sentence_embeddings

    def search(self, query_embedding, top_k=1):
        """
        Search for the most similar sentences to a query embedding
        
        Args:
            query_embedding (np.array): Query embedding
            top_k (int): Number of similar sentences to return
            
        Returns:
            list: List of indices of similar sentences
        """
        # Find the cluster closest to the query embedding
        distances = np.linalg.norm(self.cluster_centers_ - query_embedding, axis=1)
        closest_cluster_index = np.argmin(distances)

        # Find the top k closest vectors within the closest cluster
        top_k_indices = []
        for i in range(len(self.cluster_labels_)):
            if self.cluster_labels_[i] == closest_cluster_index:
                cosine_similarity_scores = np.dot(query_embedding, self.sentence_embeddings[i])/(np.linalg.norm(query_embedding)*np.linalg.norm(self.sentence_embeddings[i]))
                top_k_indices.append((i, cosine_similarity_scores))
        top_k_indices.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [index for index, _ in top_k_indices[:top_k]]

        return top_k_indices
        
class Vamana:
    def __init__(self, sentence_embeddings, n_clusters=100, n_subvectors=8, robust_prune=True):
        # Subspace dimensionality
        self.d = sentence_embeddings.shape[1] // n_subvectors
        # Number of subvectors
        self.m = n_subvectors

        # Generate random projection matrix
        self.R = np.random.randn(self.d * self.m, self.d)
        self.R /= np.linalg.norm(self.R, axis=0)

        # Project sentence embeddings onto subspaces
        self.subspace_embeddings = np.dot(sentence_embeddings, self.R)

        # Quantize subspace embeddings
        self.quantized_subspace_embeddings = np.round(self.subspace_embeddings / self.d)

        # Cluster the quantized subspace embeddings using k-means
        kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        kmeans.fit(self.quantized_subspace_embeddings)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.cluster_labels_ = kmeans.labels_
        self.sentence_embeddings = sentence_embeddings

        # Perform robust pruning
        if robust_prune:
            self.robust_prune()

    def robust_prune(self):
        # Find the clusters with the most members
        cluster_counts = np.bincount(self.cluster_labels_)
        top_k_clusters = np.argsort(cluster_counts)[-10:]

        # Remove the members of the other clusters
        for i in range(len(self.cluster_labels_)):
            if self.cluster_labels_[i] not in top_k_clusters:
                self.cluster_labels_[i] = -1

    def search(self, query_embedding, top_k=1):
        # Project query embedding onto subspaces
        query_subspace_embeddings = np.dot(query_embedding, self.R)

        # Quantize query subspace embeddings
        quantized_query_subspace_embeddings = np.round(query_subspace_embeddings / self.d)

        # Find the clusters that contain the nearest neighbors
        nearest_clusters = []
        for i in range(self.m):
            distances = np.linalg.norm(self.quantized_subspace_embeddings[:, i] - quantized_query_subspace_embeddings[i], ord=2)
            nearest_clusters.append(np.argsort(distances)[:top_k])

        # Combine the nearest clusters
        combined_nearest_clusters = set()
        for clusters in nearest_clusters:
            combined_nearest_clusters.update(clusters)

        # Find the nearest neighbors within the combined nearest clusters
        top_k_indices = []
        for i in combined_nearest_clusters:
            if self.cluster_labels_[i] != -1:
                cosine_similarity_scores = np.dot(query_embedding, self.sentence_embeddings[i])/(np.linalg.norm(query_embedding)*np.linalgnorm(self.sentence_embeddings[i]))
                top_k_indices.append((i, cosine_similarity_scores))
        top_k_indices.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [index for index, _ in top_k_indices[:top_k]]

        return top_k_indices
