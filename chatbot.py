import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data from JSON
with open("faq.json", "r") as file:
    faq_list = json.load(file)

# Extract questions and answers from the list of dicts
questions = [item["question"] for item in faq_list]
answers = [item["answer"] for item in faq_list]

# TF-IDF setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def get_bot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match_idx = similarity.argmax()
    confidence = similarity[0][best_match_idx]

    if confidence > 0.3:
        return answers[best_match_idx]
    else:
        return "Sorry, I didn't understand that. Please try asking differently."
