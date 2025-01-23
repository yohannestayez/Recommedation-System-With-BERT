# 🍷 Wine Recommendation System with BERT

A BERT-based recommendation system for wines, leveraging semantic understanding of wine descriptions and metadata to deliver personalized recommendations.

---

## 🚀 Features
- **BERT Embeddings**: Generate contextual embeddings for wine descriptions and metadata.
- **FAISS Indexing**: Efficient similarity search for scalable recommendations.
- **Hybrid Recommendations**: Combine text embeddings with categorical/numerical features.
- **Evaluation Metrics**: Precision@k, Recall@k, and custom relevance thresholds.

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yohannestayez/Recommedation-System-With-BERT.git
   cd wine-recommendation-bert
   ```

2. **Set up a conda environment**:
   ```bash
   conda create -n wine-recommendation python=3.8
   conda activate wine-recommendation
   pip install -r requirements.txt  # Install dependencies (transformers, faiss, pandas, etc.)
   ```

---


## 📊 Evaluation Results
| Metric          | Value   |
|-----------------|---------|
| **Precision@15** | 0.7133 |
| **Recall@15**    | 0.9775 |


---

## 📂 Dataset
The dataset includes wine metadata such as:
- `description`: Tasting notes and flavor profiles.
- `country`, `region`: Geographical context.
- `price`, `points`: Numerical features.
- `variety`, `winery`: Categorical attributes.

Download the dataset from [Kaggle](https://www.kaggle.com/code/kshitijmohan/wine-recommendation-system-based-on-bert/input).

