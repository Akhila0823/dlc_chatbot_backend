from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json

# Load the dataset
with open("dlc_faq_dataset.json", "r") as f:
    faq_data = json.load(f)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to compute sentence embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Precompute embeddings for all FAQ questions
faq_embeddings = [get_embedding(item["question"]) for item in faq_data]

# Function to get chatbot response
def get_bot_response(query):
    query_emb = get_embedding(query)
    scores = [cosine_similarity(query_emb, emb)[0][0] for emb in faq_embeddings]
    best_match = faq_data[scores.index(max(scores))]
    return best_match["answer"]

