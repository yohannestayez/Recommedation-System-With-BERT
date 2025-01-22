import faiss
import numpy as np
from config import SAVE_DIR

def build_faiss_index():
    """Build and save a FAISS index for embeddings."""
    embeddings = np.load(f"{SAVE_DIR}/wine_embeddings.npy").astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, f"{SAVE_DIR}/wine_index.faiss")
    return index

if __name__ == "__main__":
    build_faiss_index()