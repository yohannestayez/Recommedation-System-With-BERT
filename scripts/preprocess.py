import pandas as pd
from config import DATA_PATH, SAVE_DIR, COLUMNS, SPECIAL_TOKENS
import os

def format_structured_input(row):
    """Format a row into structured text with special tokens."""
    structured_text = (
        f"[DESC] {row['description']} "
        f"[DEG] {row['designation']} "
        f"[TITLE] {row['title']} "
        f"[VARIETY] {row['variety']} "
        f"[REGION] {row['region']} "
        f"[COUNTRY] {row['country']} "
        f"[PRICE] {row['price_range']} "
        f"[POINTS] {row['points_range']} "
        f"[WINERY] {row['winery']}"
        f"[PROVINCE] {row['province']}"
        f"[TNAME] {row['taster_name']}"
    )
    return structured_text

def preprocess_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv(DATA_PATH)
    df = df[COLUMNS] # Select relevant columns
    df["structured_input"] = df.apply(format_structured_input, axis=1)
    
    # Save processed data
    os.makedirs(SAVE_DIR, exist_ok=True)
    df.to_csv(f"{SAVE_DIR}/processed_wines.csv", index=False)
    return df

if __name__ == "__main__":
    preprocess_data()