import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import util
from collections import defaultdict


# Load csv
original_df = pd.read_csv("data.csv")

# Only extract relevant attributes
df = original_df[['product_name', 'main_category', 'Features', 'rating']].copy()

# Drop row if Features column is Null
df = df.dropna(subset=["Features"])

# Apply the above parse feature method to only include the category, feature and its description
df['parsed_feature'] = df['Features'].apply(util.parse_feature)

# Combine main_category and parsed_feature to 
df['combined_feature'] = df.apply(
    lambda row: f"{row['main_category']}: {row['parsed_feature']}", axis=1
)
combined_feature = df["combined_feature"].astype(str).tolist()


# ####################################################################
# Generate embeddings
embeddings = util.generate_embeddings(combined_feature)
vector_search = util.generate_faiss_index(embeddings)

# FastAPI app
app = FastAPI()

# Request model
class VendorQuery(BaseModel):
    software_category: str
    capabilities: List[str]


def find_matching_vendors(category: str, capabilities: List[str]):
    """
    Finds and ranks software vendors based on similarity to the given software category and required capabilities.

    This function performs the following steps:
    1. Constructs a semantic search query for each capability in the given category.
    2. Retrieves the top similar software products using vector search for each capability.
    3. Filters out products with low similarity scores (below 0.4).
    4. Aggregates similarity scores across all capabilities for each software product.
    5. Computes the average similarity score for each software and combines it with the product's rating.
    6. Returns a list of matching software vendors sorted by similarity and rating (both in descending order).

    Args:
        category (str): The software category (e.g., "CRM", "Marketing Automation").
        capabilities (List[str]): A list of features or capabilities required in the software.

    Returns:
        List[Dict[str, Union[str, float]]]: A ranked list of software products with keys:
            - "Software": Name of the software product.
            - "Similarity": Average similarity score.
            - "Rating": Product rating from the dataset.
    """
    # calculate all the feature similarity
    similar_sws = {}
    for feature in capabilities:
        # Construct a search query
        query = category + ": " + feature
        print("query", query)
        
        # Retrieve the top similar software products using vector search for each capability.
        feature_sw = util.find_similar_sw(query, vector_search, df, 10)
        
        # Aggregate similarity scores across all capabilities for each software product.
        temp = defaultdict(list)
        for k, v in feature_sw.items():
            # Set threshold to 0.4 for individual feature
            if v > 0.4:
                temp[k].append(v)
        for k, v in similar_sws.items():
            temp[k].extend(v)
        similar_sws = temp
    
    # Average similarity and combine rating
    similar_sws_list = []    
    rating_lookup = df.set_index("product_name")["rating"].to_dict()
    for k, v in similar_sws.items():
        similar_sws_list.append({"Software": k, "Similarity": sum(v)/len(v) if v else 0, "Rating": rating_lookup.get(k)})
        
    # Sort by similarity score and rating
    ranked = sorted(similar_sws_list, key=lambda x: (x["Similarity"], x["Rating"]), reverse=True)
    print("=======Recommended Softwares:========")
    for sw in ranked:
        print(f"Product: {sw['Software']}, rating: {sw['Rating']}, Similarity: {sw['Similarity']:.4f}")
    return ranked


@app.post("/vendorQualify")
def vendorQualify(query: VendorQuery):
    results = find_matching_vendors(query.software_category, query.capabilities)
    if not results:
        raise HTTPException(status_code=404, detail="No vendors matched your criteria.")
    return {"top_vendors": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
