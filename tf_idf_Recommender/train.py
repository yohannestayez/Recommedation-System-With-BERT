import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import faiss
import sys
sys.path.append('scripts')
from config import DATA_PATH, SAVE_DIR, CATEGORY_RANKS




def preprocess_data(df):
    """Clean and combine features with categorical handling"""

    text_columns = ['description', 'title', 'variety', 'region', 'country']
    df[text_columns] = df[text_columns].fillna('Unknown')
    
    # Handle categorical columns
    df['price_range'] = df['price_range'].fillna('unknown').str.lower()
    df['points_range'] = df['points_range'].fillna('unknown').str.lower()

    df['combined_text'] = (
        df['description'] + " " +
        df['title'] + " " +
        df['variety'] + " " +
        df['region'] + " " +
        df['country'] + " " +
        "PRICE_CAT: " + df['price_range'].str.lower() + " " +
        "POINTS_CAT: " + df['points_range'].str.lower()
    )
    print('Preprocessing complete.')
    return df

def train_and_save():
    """Full training pipeline"""
    # Load and preprocess
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)
    
    # Create TF-IDF pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            min_df=5,
            max_df=0.7,
            stop_words='english',
            ngram_range=(1, 2)
        )),
        ('svd', TruncatedSVD(n_components=256))
    ])

    print('Fitting TF-IDF model...')
    
    # Fit and transform
    tfidf_matrix = pipeline.fit_transform(df['combined_text'])
    tfidf_matrix = tfidf_matrix.astype('float32')
    faiss.normalize_L2(tfidf_matrix)
    print('TF-IDF model fitted.')

    # Build FAISS index
    index = faiss.IndexFlatIP(256)
    index.add(tfidf_matrix)
    print('FAISS index built.')

    # Save artifacts
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(pipeline, f"{SAVE_DIR}/tfidf_pipeline.joblib")
    df.to_csv(f"{SAVE_DIR}/processed_wines_tfidf.csv", index=False)
    faiss.write_index(index, f"{SAVE_DIR}/tfidf_index.faiss")
    np.save(f"{SAVE_DIR}/category_ranks.npy", CATEGORY_RANKS)

if __name__ == "__main__":
    train_and_save()