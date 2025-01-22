import torch
import numpy as np
from tqdm import tqdm
from model import load_bert
from config import SAVE_DIR, MAX_LENGTH, BATCH_SIZE
import pandas as pd
import os

def generate_embeddings():
    """Generate BERT embeddings for structured input."""
    tokenizer, model = load_bert()
    df = pd.read_csv(f"{SAVE_DIR}/processed_wines.csv")
    
    # Batch processing
    embeddings = []
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df["structured_input"].iloc[i:i+BATCH_SIZE].tolist()
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    # Save embeddings
    embeddings = np.array(embeddings)
    np.save(f"{SAVE_DIR}/wine_embeddings.npy", embeddings)
    return embeddings

if __name__ == "__main__":
    generate_embeddings()