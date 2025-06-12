import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data from JSON
with open("faq.json", "r") as file:
    faq_data = json.load(file)

questions = list(faq_data.keys())
answers = list(faq_data.values())

# TF-IDF setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def get_bot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match_idx = similarity.argmax()
    confidence = similarity[0][best_match_idx]

    if confidence > 0.3:  # you can tune this threshold
        return answers[best_match_idx]
    else:
        return "Sorry, I didn't understand that. Please try rephrasing your question."
