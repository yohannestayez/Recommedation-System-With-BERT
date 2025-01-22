import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
import warnings
warnings.filterwarnings("ignore")
import faiss
import torch
import sys
sys.path.append("scripts")
from model import load_bert
from config import SAVE_DIR, MAX_LENGTH
import pandas as pd


def recommend(query_text, top_k=5):
    """Query the recommendation system."""
    tokenizer, model = load_bert()
    df = pd.read_csv(f"{SAVE_DIR}/processed_wines.csv")
    
    # Tokenize query
    inputs = tokenizer(
        query_text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
    
    # Load FAISS index
    index = faiss.read_index(f"{SAVE_DIR}/wine_index.faiss")
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]

if __name__ == "__main__":
    query = input("Enter a wine description: ")
    recommendations = recommend(query)
    print(recommendations[["title", "variety","country", "price_range", "description"]])