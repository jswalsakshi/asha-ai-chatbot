# utils/vector_store.py
import faiss
import numpy as np

# Load embeddings
embeddings = np.load('data/job_embeddings.npy')

# Create FAISS index
dim = embeddings.shape[1]  # vector size
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index
faiss.write_index(index, 'data/faiss_index.bin')
