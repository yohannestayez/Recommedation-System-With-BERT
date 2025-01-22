from transformers import BertTokenizer, BertModel
from config import SPECIAL_TOKENS
import torch

def load_bert():
    """Load BERT tokenizer and model with special tokens."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    
    model = BertModel.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))  # Update for new tokens
    model = model.to(torch.device("cuda"))
    return tokenizer, model