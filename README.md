```markdown
# 🤖 Smart NLP AI Chatbot 💬

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-329555?logo=python&logoColor=white)

**Project Level:** Intermediate (Level 2)  
**Domain:** Artificial Intelligence & Machine Learning  
**Institution:** Cognevance Technologies  

---

## 📑 Project Overview
An end-to-end Artificial Intelligence chatbot built using Natural Language Processing (NLP) and Deep Learning. This project takes raw user text input, processes it using advanced NLP techniques, predicts the user's underlying intent using a trained Artificial Neural Network (ANN), and replies with contextually appropriate responses via a modern web interface.

---

## 📸 Interactive Web Interface
*A fully interactive, responsive chat interface featuring custom avatars and session state memory.*


## 🚀 Key Features
* **Custom Deep Learning Model:** Built and trained a Sequential Neural Network using TensorFlow/Keras to classify text intents accurately.
* **NLP Text Processing:** Utilized the Natural Language Toolkit (NLTK) for complex word tokenization, lemmatization (WordNet), and TF-IDF vectorization.
* **Modern Web UI:** Deployed a highly intuitive frontend using Streamlit to simulate a real-world messaging application.
* **Customizable Dataset:** Trained on a custom CSV dataset that maps user utterances to specific intent categories and corresponding bot responses.

---

## 🛠️ Tech Stack
* **Core Language:** Python 3
* **Deep Learning:** TensorFlow & Keras 
* **Natural Language Processing:** NLTK (Natural Language Toolkit)
* **Data Processing:** Pandas, NumPy, Scikit-Learn (Label Encoding)
* **Frontend Framework:** Streamlit 

---

## 🗂️ Repository Structure
```text
cognevance_NLP_Chatbot/
├── app.py                  # Main Streamlit web application script
├── chatbot_dataset.csv     # Training data (intents, phrases, responses)
├── chatbot_model.keras     # Saved TensorFlow neural network model
├── tokenizer.pkl           # Saved text vectorizer object
├── encoder.pkl             # Saved label encoder for intent mapping
└── README.md               # Project documentation
