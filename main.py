# prompt: create vector database from scratch
from sentence_transformers import SentenceTransformer
from index import Flat, IVF, HNSW, ProductQuantization, ScalarQuantization, Vamana
from helper import timeit_wrapper, memory_usage
import numpy as np
import os
import pickle


class SimpleVectorDatabase:
    @timeit_wrapper
    @memory_usage
    def __init__(
        self, input_sentences, index_method="hnsw", embedding_model="all-MiniLM-L6-v2"
    ):
        """
        Simple Vector Database

        Args:
        input_sentences (list): list of sentences in str.
        index_method (str): index method for vectors. Default is hnsw.
                            methods: flat, ivf, hnsw, pq, sq, diskann (not supported yet).
        embedding_model (str): embedding model from sentence-transformers.
                                Default is all-MiniLM-L6-v2.

        """
        if input_sentences is None:
            raise ValueError("Please input a list of sentences.")
        if not isinstance(input_sentences, list):
            raise ValueError("Input sentences should be a list.")
        if not all(isinstance(sentence, str) for sentence in input_sentences):
            raise ValueError("All elements in the list should be strings.")

        if os.path.exists("./__cache__/embedding_models/"):
            self.model = SentenceTransformer(
                embedding_model, cache_folder="./__cache__/embedding_models/"
            )
        else:
            self.model = SentenceTransformer(embedding_model)
        if os.path.exists("./__cache__/data/test_db.pkl"):
            self.load_database("./__cache__/data/test_db.pkl")

        else:
            # loading the sentence transformer model
            # TODO: These two steps take a lot of time. Figure out a way to cache them.
            if not os.path.exists("./__cache__/embeddings"):
                os.makedirs("./__cache__/embeddings")
            if os.path.exists("./__cache__/embeddings/sentence_embeddings.npy"):
                self.sentence_embeddings = np.load(
                    "./__cache__/embeddings/sentence_embeddings.npy"
                )
            else:
                self.sentence_embeddings = self.model.encode(
                    input_sentences
                )  # TODO: memory usage is high. Figure why.

                # caching the embeddings
                np.save(
                    "./__cache__/embeddings/sentence_embeddings.npy",
                    self.sentence_embeddings,
                )

            self.input_sentences = input_sentences
            self.index_method = index_method

        if index_method == "flat":
            self.index = Flat(self.sentence_embeddings)
        elif index_method == "ivf":
            self.index = IVF(self.sentence_embeddings)
        elif index_method == "hnsw":
            self.index = HNSW(self.sentence_embeddings, input_sentences)
        elif index_method == "pq":
            self.index = ProductQuantization(self.sentence_embeddings)
        elif index_method == "sq":
            self.index = ScalarQuantization(self.sentence_embeddings)
        elif index_method == "diskann":
            # self.index = Vamana(self.sentence_embeddings)
            raise NotImplementedError("DiskANN is not supported yet.")
        else:
            raise ValueError(
                "Index method not supported. Please choose from flat, ivf, hnsw, pq, sq, diskann."
            )

    @timeit_wrapper
    @memory_usage
    def search(self, query, top_k=1):
        """
        Search for the most similar sentence in the database.

        Args:
        query (str): query sentence.
        top_k (int): number of similar sentences to return. Default is 1.

        Returns:
        list: list of indices of similar sentences.
        """
        query_embedding = self.model.encode(query)
        top_k_indices = self.index.search(query_embedding, top_k)
        top_k_sentences = [self.input_sentences[i] for i in top_k_indices]
        return top_k_sentences

    def add(self, sentence):
        """
        Add a new sentence to the database.

        Args:
        sentence (str): sentence to add.
        """
        if not isinstance(sentence, str):
            raise ValueError("Input should be a string.")
        self.input_sentences.append(sentence)
        new_embedding = self.model.encode(sentence)
        self.sentence_embeddings = np.vstack([self.sentence_embeddings, new_embedding])

        # update the index
        if self.index_method == "flat":
            self.index = Flat(self.sentence_embeddings)
        elif self.index_method == "ivf":
            self.index = IVF(self.sentence_embeddings)
        elif self.index_method == "hnsw":
            self.index = HNSW(self.sentence_embeddings, self.input_sentences)
        elif self.index_method == "pq":
            self.index = ProductQuantization(self.sentence_embeddings)
        elif self.index_method == "sq":
            self.index = ScalarQuantization(self.sentence_embeddings)
        elif self.index_method == "diskann":
            # self.index = Vamana(self.sentence_embeddings)
            raise NotImplementedError("DiskANN is not supported yet.")
        else:
            raise ValueError(
                "Index method not supported. Please choose from flat, ivf, hnsw, pq, sq, diskann."
            )

    def remove(self, sentence):
        """
        Remove a sentence from the database.

        Args:
        sentence (str): sentence to remove.
        """
        if not isinstance(sentence, str):
            raise ValueError("Input should be a string.")
        if sentence not in self.input_sentences:
            raise ValueError("Sentence not found in the database.")
        index = self.input_sentences.index(sentence)
        self.input_sentences.remove(sentence)
        self.sentence_embeddings = np.delete(self.sentence_embeddings, index, axis=0)

        # update the index
        if self.index_method == "flat":
            self.index = Flat(self.sentence_embeddings)
        elif self.index_method == "ivf":
            self.index = IVF(self.sentence_embeddings)
        elif self.index_method == "hnsw":
            self.index = HNSW(self.sentence_embeddings, self.input_sentences)
        elif self.index_method == "pq":
            self.index = ProductQuantization(self.sentence_embeddings)
        elif self.index_method == "sq":
            self.index = ScalarQuantization(self.sentence_embeddings)
        elif self.index_method == "diskann":
            # self.index = Vamana(self.sentence_embeddings)
            raise NotImplementedError("DiskANN is not supported yet.")
        else:
            raise ValueError(
                "Index method not supported. Please choose from flat, ivf, hnsw, pq, sq, diskann."
            )

    def get(self, index):
        """
        Get a sentence from the database.

        Args:
        index (int): index of the sentence.

        Returns:
        str: sentence at the index.
        """
        if not isinstance(index, int):
            raise ValueError("Index should be an integer.")
        if index < 0 or index >= len(self.input_sentences):
            raise ValueError("Index out of bounds.")
        return self.input_sentences[index]

    def save_database(self, file_name="sentences.pkl", force=False):
        """
        Save the database to a file.

        Args:
        file_name (str): name of the file to save. Type should be .pkl.
        """
        DATA_DIR = "./__cache__/data"
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        if force is False:
            if os.path.exists(DATA_DIR + "/" + file_name):
                raise ValueError(
                    "File already exists. If you wish to overwrite, please add force=True."
                )
        else:
            data = {
                "input_sentences": self.input_sentences,
                "sentence_embeddings": self.sentence_embeddings.tolist(),
                "index_method": self.index_method,
            }
            with open(DATA_DIR + "/" + file_name, "wb") as f:
                pickle.dump(data, f)

    def load_database(self, path="./__cache__/data/sentences.pkl"):
        """
        Load the database from a file.

        Args:
            path (str): path to load the database.
        """

        if not os.path.exists(path):
            raise ValueError("File not found.")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.input_sentences = data["input_sentences"]
            self.sentence_embeddings = np.array(data["sentence_embeddings"])
            self.index_method = data["index_method"]
        except Exception as e:
            raise ValueError("Error loading the data: " + str(e))

    def __len__(self):
        return len(self.input_sentences)

    def __repr__(self):
        return f"SimpleVectorDatabase(input_sentences={self.input_sentences}, index_method={self.index_method})"

    def __str__(self):
        return f"SimpleVectorDatabase with {len(self.input_sentences)} sentences and index method {self.index_method}."
