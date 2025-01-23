import os
import pandas as pd
import numpy as np
import joblib
import faiss
import sys
sys.path.append('scripts')
from config import SAVE_DIR

class TFIDFRecommender:
    def __init__(self):
        self.pipeline = joblib.load(f"{SAVE_DIR}/tfidf_pipeline.joblib")
        self.df = pd.read_csv(f"{SAVE_DIR}/processed_wines.csv")
        self.category_ranks = np.load(
            f"{SAVE_DIR}/category_ranks.npy", 
            allow_pickle=True
        ).item()
        self.index = faiss.read_index(f"{SAVE_DIR}/tfidf_index.faiss")

    def _filter_by_category(self, results, column, category):
        """Filter results based on ordinal categories"""
        if not category:
            return results
            
        target_rank = self.category_ranks.get(category.lower(), 0)
        valid_cats = [
            cat for cat, rank in self.category_ranks.items() 
            if rank >= target_rank
        ]
        return results[results[column].str.lower().isin(valid_cats)]

    def recommend(self, query, k=10, price_cat=None, points_cat=None):
        """Generate recommendations with categorical filters"""
        # Transform query
        query_vec = self.pipeline.transform([query]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search index
        _, indices = self.index.search(query_vec, k=100)
        results = self.df.iloc[indices[0]]
        
        # Apply categorical filters
        if price_cat:
            results = self._filter_by_category(results, 'price_range', price_cat)
        if points_cat:
            results = self._filter_by_category(results, 'points_range', points_cat)
            
        return results.drop_duplicates('title').head(k)

    def evaluate(self, test_queries, k=5):
        """Evaluate system performance"""
        metrics = {'precision': [], 'recall': []}
        
        for query in test_queries:
            results = self.recommend(query, k=15)
            
            # Simulate relevant items (points >= medium)
            relevant = self._filter_by_category(results, 'points_range', 'medium')
            
            # Calculate metrics
            top_k = relevant.head(k)
            metrics['precision'].append(len(top_k)/k)
            metrics['recall'].append(len(top_k)/len(relevant) if len(relevant) >0 else 0)
            
        return {k: np.mean(v) for k, v in metrics.items()}

if __name__ == "__main__":
    # Initialize recommender
    recommender = TFIDFRecommender()
    
    # Example usage
    results = recommender.recommend(
        "Fruity red wine with berry notes",
        price_cat='Medium',
        points_cat='High'
    )
    
    print("Top Recommendations:")
    print(results[['title', 'country','price_range', 'points_range']].head(10))
    
    # Example evaluation
    test_queries = [
        "Dry white wine from France",
        "Bold Cabernet with tannins",
        "Award-winning Chardonnay"
    ]
    metrics = recommender.evaluate(test_queries)
    print("\nEvaluation Metrics:")
    print(f"Precision@5: {metrics['precision']:.4f}")
    print(f"Recall@5: {metrics['recall']:.4f}")