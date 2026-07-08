# ✨ Smart NLP AI Chatbot

An end-to-end Artificial Intelligence chatbot built using Natural Language Processing (NLP) and Deep Learning. This project takes user text input, processes it, predicts the user's intent using a trained neural network, and replies with an appropriate response via a modern web interface.

## 🚀 Features
* **Custom Deep Learning Model:** Built a Sequential Neural Network using TensorFlow/Keras to classify text intents.
* **NLP Text Processing:** Utilized NLTK for word tokenization, lemmatization (WordNet), and TF-IDF vectorization.
* **Modern Web UI:** Deployed a fully interactive, responsive chat interface using Streamlit, featuring custom avatars and session state memory.
* **Customizable Dataset:** Trained on a custom CSV dataset mapping user utterances to specific intents and bot responses.

## 🛠️ Tech Stack
* **Python 3**
* **TensorFlow & Keras** (Deep Learning)
* **NLTK** (Natural Language Toolkit)
* **Pandas & NumPy** (Data Handling)
* **Scikit-Learn** (Label Encoding & Splitting)
* **Streamlit** (Frontend Web Framework)

## 📁 Repository File Structure
* `app.py`: The main Streamlit web application script.
* `chatbot_dataset.csv`: The training dataset containing intents, user phrases, and bot responses.
* `chatbot_model.keras`: The saved, pre-trained TensorFlow neural network model.
* `tokenizer.pkl`: The saved text vectorizer to convert words to numbers.
* `encoder.pkl`: The saved label encoder mapping intent categories.
* *(Optional: Add a screenshot of your web app here!)*

## 💻 How to Run Locally

If you want to run this chatbot on your own machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
