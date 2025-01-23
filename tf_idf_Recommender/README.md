#  TF-IDF Recommendation System

A lightweight recommendation system using TF-IDF and categorical features to suggest wines based on descriptions, metadata, and user preferences.



## Features
- **TF-IDF with Categorical Integration**: Combines text descriptions with price/points categories.
- **FAISS Indexing**: Fast similarity search for 100k+ wines.
- **Category Filtering**: Support for ordinal filters (Low/Medium/High).




## Folder Structure
```
tf_idf_Recommender/
├── train.py               # Trains TF-IDF model and builds FAISS index
├── tfidf_recommender.py   # Recommendation logic with categorical filters
└── __init__.py            # Package initialization
```


