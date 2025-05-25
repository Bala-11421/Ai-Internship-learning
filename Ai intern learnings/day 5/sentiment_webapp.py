import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Sample training data (replace with your Day 4 model)
def create_sample_model():
    reviews = [
        "This movie was great! I loved it.",
        "Terrible movie, waste of time.",
        "The acting was superb.",
        "I fell asleep, so boring.",
        "One of the best movies this year!",
        "The worst movie ever made."
    ]
    sentiments = [1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative
    
    df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
    df['processed'] = df['review'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed'])
    model = LogisticRegression()
    model.fit(X, df['sentiment'])
    
    return vectorizer, model

# Load model
vectorizer, model = create_sample_model()

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and click Predict to analyze sentiment")

user_input = st.text_area("Movie Review:", "This movie was...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        # Preprocess and predict
        processed_text = preprocess_text(user_input)
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)
        proba = model.predict_proba(text_vector)[0]
        
        # Display results
        if prediction[0] == 1:
            st.success(f"✅ POSITIVE (Confidence: {proba[1]:.0%})")
            st.balloons()
        else:
            st.error(f"❌ NEGATIVE (Confidence: {proba[0]:.0%})")
        
        # Optional: Show explanation
        with st.expander("Analysis Details"):
            st.write("Processed Text:", processed_text)
            st.write("Prediction Probabilities:", proba)
