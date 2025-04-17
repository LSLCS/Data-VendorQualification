## 🧠 Vendor Qualification System

A lightweight semantic search API that helps match and rank software vendors based on user-specified software categories and capabilities using sentence embeddings and FAISS for fast vector similarity search.

### 🔍 Overview

This project takes a dataset of software products (including software name, features, categories, and ratings), processes feature descriptions using NLP, and enables users to query software vendors that best match a given category and list of required capabilities.

### 📦 Components

- `vendorQualify.py` – Main FastAPI app that handles incoming queries and returns top matching vendors.
- `util.py` – Utility functions for parsing features, generating embeddings, building a FAISS index, and similarity search.
- `vector_search.py` – Lightweight wrapper for FAISS vector indexing and searching.
- `data.csv` – Source data with product details, categories, features, and ratings.

---

### 🚀 How It Works

1. Parses the `Features` column (stored as JSON strings) into readable text.
2. Combines features and categories into embedding-ready text.
3. Generates sentence embeddings using `all-MiniLM-L6-v2` via `sentence-transformers`.
4. Builds a FAISS index for efficient similarity search.
5. Accepts queries through a FastAPI endpoint and ranks vendors based on:
   - Average similarity score across all required capabilities.
   - Product rating if same similarity score

---

### 🧪 Sample API Usage

**Endpoint**: `POST /vendorQualify`

**Request Body**:
```json
{
  "software_category": "CRM",
  "capabilities": [
    "Lead Management",
    "Automated Email Campaigns"
  ]
}
```

**Response**:
```json
{
  "top_vendors": [
    {
      "Software": "Efficy CRM",
      "Similarity": 0.73,
      "Rating": 4.2
    },
    ...
  ]
}
```

---

### 🛠️ Setup Instructions

#### 1. Clone the repo

```bash
git clone https://github.com/yourusername/vendor-qualification
cd vendor-qualification
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Dependencies**:
> - `fastapi`
> - `uvicorn`
> - `pandas`
> - `numpy`
> - `sentence-transformers`
> - `faiss-cpu` or `faiss-gpu`

#### 3. Prepare the dataset

Ensure `data.csv` is placed in the root directory or update the path in `vendorQualify.py`.


#### 4. Run the API server

Start the FastAPI server in one of the following ways:

**Option 1** – Using `python` directly (only works if `vendorQualify.py` includes the `__main__` block):

```bash
python vendorQualify.py
```

**Option 2** – Using `uvicorn`:

```bash
uvicorn vendorQualify:app --reload
```

Once the server is running, visit:

 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) – interactive API testing interface  

Use the `/vendorQualify` POST endpoint to submit a software category and a list of capabilities to get the top matching software vendors.


## 🚀 Future Improvements

- **Dockerization**: Containerize the application using Docker for easier deployment and environment consistency.
- **Testing**: Add unit and integration tests using `pytest` or `unittest` to ensure robustness and reliability of core functionalities.
- **Enhanced Querying**: Allow users to optionally specify other metadata columns (e.g., price, deployment type, or integration support) to influence similarity scoring and ranking.
- **Configurable Thresholds**: Make similarity thresholds and ranking weights configurable via request parameters or a config file.
- **UI Integration**: Build a simple frontend or Swagger UI enhancements for better interaction with the API.
- **Persist FAISS Index**: Persist the vector index to disk and load it on startup to avoid regenerating embeddings and index on every run.
