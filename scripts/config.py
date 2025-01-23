# Paths and constants
DATA_PATH = "../Data/Processed/wine_data_processed.csv"
SAVE_DIR = "../Data/Processed"
SPECIAL_TOKENS = [
    "[DESC]", "[DEG]","[TITLE]", "[VARIETY]", "[REGION]",
    "[COUNTRY]", "[PRICE]", "[POINTS]", "[WINERY]", 
    "[PROVINCE]", "[TNAME]"
]
COLUMNS = [
    "description","designation", "title", "variety", "region",
    "country", "price_range", "points_range", "winery", 
    "province","taster_name"
]
MAX_LENGTH = 512  # BERT's token limit
BATCH_SIZE = 16    # Batch size for training


# Evaluation parameters
TEST_SIZE = 0.2         # Split for train/test
METRIC_TOP_K = [10, 15]  # Evaluate for top 10 and top 15