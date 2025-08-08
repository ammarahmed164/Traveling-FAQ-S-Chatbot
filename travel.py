import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

faq_data = [
    {"question": "Do I need a passport to travel?", "answer": "Yes, a passport is required for international travel."},
    {"question": "What is the best time to visit Japan?", "answer": "Spring (March-May) and Fall (September-November) are best."},
    {"question": "How can I book cheap flights?", "answer": "Use comparison websites and book in advance for cheaper prices."},
    {"question": "Do I need a visa for the UK?", "answer": "It depends on your nationality. Check the UK government website."},
    {"question": "What should I pack for a beach vacation?", "answer": "Pack sunscreen, swimwear, flip-flops, and light clothes."},
    {"question": "Can I travel with pets?", "answer": "Yes, but you must check airline rules and pet travel guidelines."}
]

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]  # remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

questions = []
for faq in faq_data:
    questions.append(faq['question'])
preprocessed_questions = []
for q in questions:
    preprocessed_questions.append(preprocess(q))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

def get_answer(user_question):
    user_question_clean = preprocess(user_question)
    user_vector = vectorizer.transform([user_question_clean])
    
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    best_match_idx = similarities.argmax()
    score = similarities[0][best_match_idx]

    if score > 0.3:  
        return faq_data[best_match_idx]['answer']
    else:
        return "Sorry, I don't understand that question."

# Streamlit UI
st.set_page_config(page_title="Travel Q&A Chatbot", page_icon="🛫", layout="centered")

st.title("🛫 Travel Q&A Chatbot")
st.write("Ask me any travel-related question! Type below and press **Enter**.")

user_input = st.text_input("Your Question:", "")
if user_input:
    answer = get_answer(user_input)
    st.markdown(f"**Bot:** {answer}")
