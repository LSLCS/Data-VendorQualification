import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from vector_search import VectorSearch
import json

# ####################################################################
# Parse feature column
def parse_feature(feature_str):
    """
    Parses a JSON-formatted string of feature data and converts it into a flat text string.

    The input string is expected to contain a list of categories, where each category includes a 
    list of features. Each feature has a name and a description. The function concatenates the 
    category name followed by each feature's name and description into a single string.

    Args:
        feature_str (str): A JSON-formatted string representing categorized features.

    Returns:
        str: A single string with all categories and features formatted as:
             "Category1: Feature1 - Description1; Feature2 - Description2 | Category2: ...".
    """
    content = json.loads(feature_str)
    all_features = ""
    for category in content:
        all_features += category.get('Category', 'Unknown') + ": "
        # print(category['Category'])
        for feature in category['features']:
            name = feature.get('name', '')
            desc = feature.get('description', '')
            if name and desc:
                all_features += f"{name} - {desc}"
        all_features += " | "
    return all_features


# ####################################################################
# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ####################################################################
# Generate embeddings
def generate_embeddings(job_descriptions, save_path="other/job_embeddings.npy", as_tensor=False):
    """
    Generate embeddings for job descriptions and save them.
    
    :param job_descriptions: List of job descriptions
    :param save_path: Path to save the embeddings
    :param as_tensor: Whether to return embeddings as PyTorch tensors
    :return: Generated embeddings
    """
    print("Generating embeddings...")
    embeddings = model.encode(job_descriptions, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    # Save embeddings to a file for later use 
    np.save(save_path, embeddings)
    print(f"embeddings saved to {save_path}")

    return embeddings

# ####################################################################
# Generate faiss index
def generate_faiss_index(embeddings, index_path="other/job_index.faiss"):
    """
    Build a FAISS index from embeddings and save it to a file.

    :param embeddings: NumPy array of embeddings (shape: [n_samples, dim])
    :param index_path: Path where the FAISS index will be saved
    :return: VectorSearch object with the index
    """
    # Determine embedding dimension
    dimension = embeddings.shape[1]
    
    # Initialize FAISS vector search index
    vector_search = VectorSearch(dimension)
    
    # Add embeddings
    vector_search.add_embeddings(embeddings)
    print(f"Indexed {len(embeddings)} items.")
    
    # Save index to file
    vector_search.save_index(index_path)
    print(f"FAISS index saved to {index_path}")
    
    return vector_search


# ####################################################################
# Perform vector search and calculate similarity
def find_similar_sw(query_text, vector_search, df, top_k=10):
    """
    Finds the top-k software products most similar to a given query based on vector embeddings.

    Args:
        query_text (str): The input text describing the desired software features or requirements.
        vector_search: A FAISS or similar vector search index used to perform nearest-neighbor search.
        df (pd.DataFrame): A DataFrame containing software metadata, including 'product_name' and 'rating'.
        top_k (int, optional): The number of most similar software results to return. Defaults to 10.

    Returns:
        results : dictionary, product_name: Similarity Score

    Notes:
    -----
    - Uses a precomputed embedding model to embed the query.
    - Performs approximate nearest neighbor search over the software embedding index.
    - Converts cosine distance to similarity using the formula: `1 / (1 + distance)`.
    - Skips the first result (index 0) assuming it's the query itself.
    """
    query_embedding = model.encode([query_text]).astype("float32")
    distances, indices = vector_search.search(query_embedding, top_k + 1)
    results = {}
    for i in range(1, top_k + 1):
        sw_id = indices[0][i]
        similarity = float(1 / (1 + distances[0][i]) )
        results[df.iloc[sw_id]["product_name"]] = similarity
    return results
