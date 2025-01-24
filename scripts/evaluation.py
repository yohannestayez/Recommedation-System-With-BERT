import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict
import faiss
from tqdm import tqdm
import pandas as pd
import numpy as np
from config import *
from sklearn.model_selection import train_test_split

def load_data():
    """Load processed data with embeddings and clean points_range."""
    df = pd.read_csv(f"{SAVE_DIR}/processed_wines.csv")
    
    # Clean points_range column
    df['points_range'] = (
        df['points_range']
        .fillna('unknown')  # Handle missing values
        .astype(str)        # Ensure string type
        .str.lower()        # Convert to lowercase
    )
    
    embeddings = np.load(f"{SAVE_DIR}/wine_embeddings.npy")
    return df, embeddings

def train_test_split_data(df):
    """Split data into train/test sets."""
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    return train_df, test_df

class Evaluator:
    def __init__(self):
        self.df, self.embeddings = load_data()
        self.train_df, self.test_df = train_test_split_data(self.df)
        self.train_embeddings = self.embeddings[self.train_df.index]
        self.test_embeddings = self.embeddings[self.test_df.index]
        self._build_faiss_index()

    def _build_faiss_index(self):
        """Build FAISS index on training embeddings."""
        dimension = self.train_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.train_embeddings.astype('float32'))

    def _calculate_relevance(self, test_category, retrieved_categories):
        """Check relevance based on category ranks."""
        test_rank = CATEGORY_RANKS.get(test_category.lower(), 0)
        retrieved_ranks = [
            CATEGORY_RANKS.get(str(cat).lower(), 0)  # Handle any remaining non-string values
            for cat in retrieved_categories
        ]
        return np.array([rank >= test_rank for rank in retrieved_ranks], dtype=int)

    def evaluate(self):
        """Calculate Precision@k and Recall@k."""
        results = {f"Precision@{k}": [] for k in METRIC_TOP_K}
        results.update({f"Recall@{k}": [] for k in METRIC_TOP_K})

        # Iterate over test set with progress bar
        for test_embedding, test_row in tqdm(zip(self.test_embeddings, self.test_df.itertuples()), 
                                           total=len(self.test_df),
                                           desc="Evaluating"):
            # Search FAISS index
            distances, indices = self.index.search(test_embedding.reshape(1, -1).astype('float32'), 
                                                 max(METRIC_TOP_K))
            retrieved_wines = self.train_df.iloc[indices[0]]

            # Get test wine's category (already cleaned during load_data)
            test_category = test_row.points_range

            # Calculate relevance for retrieved wines
            relevance = self._calculate_relevance(
                test_category,
                retrieved_wines['points_range'].values
            )

            # Compute metrics for each k
            for k in METRIC_TOP_K:
                top_k_relevance = relevance[:k]
                num_relevant = np.sum(top_k_relevance)

                # Precision@k: Relevant items in top-k / k
                precision = num_relevant / k
                results[f"Precision@{k}"].append(precision)

                # Recall@k: Relevant items in top-k / total relevant items
                total_relevant = np.sum(relevance)
                recall = num_relevant / total_relevant if total_relevant > 0 else 0
                results[f"Recall@{k}"].append(recall)

        # Aggregate results
        for metric in results:
            results[metric] = np.mean(results[metric])
        return results

if __name__ == "__main__":
    evaluator = Evaluator()
    metrics = evaluator.evaluate()
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")