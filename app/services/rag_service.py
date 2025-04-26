import faiss
import numpy as np
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from sentence_transformers import SentenceTransformer


HUGGINGFACE_API_URL = os.environ["HUGGINGFACE_API_URL"]
HUGGINGFACE_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Where you cloned the models
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "utils", "models", "bge-small-en-v1.5")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- LOAD MODELS ---
# 1. Load embedding model (local)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

def generate_response_huggingface(query, context):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
    payload = {
        "inputs": f"Question: {query}\nContext: {context}\nAnswer:"
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, verify=False)
    response_json = response.json()
    
    # Assuming the response contains a "generated_text" field
    return response_json[0]['generated_text']


# Absolute path to the local model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "utils", "models", "bge-small-en-v1.5")

# Paths to index and chunks
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(DATA_DIR, "job_chunks.npy")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load text chunks (job listings)
job_chunks = np.load(CHUNKS_PATH, allow_pickle=True)

# Load local embedding model
model = SentenceTransformer(MODEL_PATH)

def semantic_search(query, top_k=3):
    """
    Returns top_k most semantically similar job chunks for the user query.
    """
    query_vector = model.encode([query])
    indices = index.search(np.array(query_vector), top_k)[1]
    results = [job_chunks[i] for i in indices[0]]
    return results

query = "Looking for cooking job"
results = semantic_search(query)

for i, res in enumerate(results, 1):
    print(f"{i}. {res}")


# --- TEST ---
if __name__ == "__main__":
    query = "Are there any remote frontend jobs?"
    context = "\n".join(semantic_search(query))
    print("Context:", context)
    print("Response from Hugging Face API:", generate_response_huggingface(query, context))

