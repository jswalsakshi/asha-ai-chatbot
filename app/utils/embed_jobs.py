from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import logging

# Setup paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bge-small-en-v1.5")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
JOB_LISTING_FILE = os.path.join(DATA_DIR, "job_listing_data.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "job_embeddings.npy")
CHUNKS_FILE = os.path.join(DATA_DIR, "job_chunks.npy")  # <- NEW

def generate_embeddings():
    try:
        # 1. Load job listing data
        df = pd.read_csv(JOB_LISTING_FILE)
        texts = df.apply(
            lambda row: f"Job: {row['job_title']} at {row['company']} in {row['location']}. Skills: {row.get('skills', '')}",
            axis=1
        ).tolist()

        # 2. Load local model
        model = SentenceTransformer(MODEL_PATH)

        # 3. Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True)

        # 4. Save embeddings and chunks
        np.save(EMBEDDINGS_FILE, embeddings)
        np.save(CHUNKS_FILE, texts)  # <- SAVE CHUNKS HERE

        logging.info(f"✅ Embeddings saved to {EMBEDDINGS_FILE}")
        logging.info(f"✅ Text chunks saved to {CHUNKS_FILE}")

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_embeddings()
