# Recommendation System with BERT

A BERT-based recommendation system for wines, leveraging semantic understanding of wine descriptions and metadata to deliver personalized recommendations.

---

## Features
- **BERT Embeddings**: Generate contextual embeddings for wine descriptions and metadata.
- **FAISS Indexing**: Efficient similarity search for scalable recommendations.
- **Hybrid Recommendations**: Combine text embeddings with categorical/numerical features.
- **Evaluation Metrics**: Precision@k, Recall@k, and custom relevance thresholds.

---

## Project Structure

The project is organized as follows:

```
├── .github/
│    ├── workflows
│        └── unittest.yml          # GitHub Actions workflow for running unit tests
├── Data/                          # Directory for storing datasets
├── notebooks/
│   ├── Data_Preprocessing.ipynb   # Jupyter notebook for data preprocessing
│   └── Recommendation_sys.ipynb   # Jupyter notebook for building the recommendation system
├── scripts/
│   ├── __init__.py                # Initialization file for the scripts module
│   ├── config.py                  # Configuration settings for the project
│   ├── embeddings.py              # Script for generating BERT embeddings
│   ├── evaluation.py              # Script for evaluating the BERT model
│   ├── faiss_index.py             # Script for creating and querying the FAISS index
│   ├── model.py                   # Script for defining the BERT recommendation model
│   ├── preprocess.py              # Script for preprocessing the dataset
├── src/
│    ├── __init__.py               # Initialization file for the src module
│    └── main.py                   # Main script to run the recommendation system
├── tf_idf_Recommender/
│   ├── __init__.py                # Initialization file for the tf_idf_Recommender module
│   ├── tfidf_recommender.py       # Script for TF-IDF based recommendation system
│   ├── train.py                   # Script for training the TF-IDF model
│   └── README.md                  # README file for the TF-IDF recommender module
├── README.md                      # Project README file
├── .gitignore                     # Git ignore file
└── requirements.txt               # List of dependencies
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

