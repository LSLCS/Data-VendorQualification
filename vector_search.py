import faiss
import numpy as np

class VectorSearch:
    def __init__(self, dim, index_file=None):
        """
        Initialize a FAISS index.
        If an index file is provided, it loads the existing index.
        """
        self.dim = dim
        # Create an index (L2 nearest neighbor)
        self.index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean)
        
        if index_file:
            self.load_index(index_file)

    def add_embeddings(self, embeddings):
        """
        Add embeddings to the FAISS index.
        """
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        """
        Search for the top-k most similar vectors.
        Returns distances and indices of matches.
        """
        # query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save_index(self, file_path):
        """
        Save the FAISS index to a file.
        """
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path):
        """
        Load a FAISS index from a file.
        """
        self.index = faiss.read_index(file_path)