# Paths and constants
DATA_PATH = "Data/Processed/wine_data_processed.csv"
SAVE_DIR = "Data/Processed"
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
BATCH_SIZE = 16    # Adjust based on GPU memory


#country	description	designation		province	taster_name	taster_twitter_handle	title	variety	winery	region	price_range	points_range