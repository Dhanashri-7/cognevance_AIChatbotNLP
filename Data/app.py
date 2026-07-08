import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pickle
import random

# --- PAGE CONFIGURATION ---
# This changes the browser tab title and expands the layout
st.set_page_config(page_title="AI Assistant", page_icon="✨", layout="centered")

# Load all the saved files
@st.cache_resource
def load_ai_components():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    model = load_model("chatbot_model.keras")
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    df = pd.read_csv('/content/chatbot_dataset.csv')
    return model, tokenizer, encoder, df

model, tokenizer, encoder, df = load_ai_components()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return " ".join(lemmatized_tokens)

# --- CUSTOM CSS ---
# Adds a custom font color to the header and a light background
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 About this Bot")
    st.write("This is a custom NLP chatbot built with **TensorFlow**, **NLTK**, and **Streamlit**.")
    st.divider()
    st.write("💡 **Try asking:**")
    st.write("- *Hello!*")
    st.write("- *Where is my order?*")
    st.write("- *I need help.*")

# --- MAIN CHAT UI ---
st.title("✨ My Smart NLP Assistant")
st.caption("🚀 Powered by Deep Learning")

# Initialize chat history with a default welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I am your AI assistant. How can I help you today?"}
    ]

# Display chat messages with custom avatars!
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input box
if user_input := st.chat_input("Type your message here..."):
    # 1. Show user message with avatar
    st.chat_message("user", avatar="👤").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. AI Prediction
    cleaned_text = preprocess_text(user_input)
    input_vector = tokenizer.texts_to_matrix([cleaned_text], mode='tfidf')
    predicted_probs = model.predict(input_vector, verbose=0)
    predicted_label_index = np.argmax(predicted_probs, axis=1)[0]
    predicted_intent = encoder.inverse_transform([predicted_label_index])[0]
    
    # 3. Get Bot response
    possible_responses = df[df['Entities'] == predicted_intent]['Bot Response'].dropna().tolist()
    bot_reply = random.choice(possible_responses) if possible_responses else "I'm sorry, I don't understand."
    
    # 4. Show bot message with avatar
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
