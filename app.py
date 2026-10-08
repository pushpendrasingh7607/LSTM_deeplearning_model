import os
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_model (1).h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN_PATH = os.path.join(BASE_DIR, "max_len.pkl")


@st.cache_resource
def load_resources():
    """Load the trained model and preprocessing artifacts from the same folder."""
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(MAX_LEN_PATH, "rb") as f:
        max_len = int(pickle.load(f))

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()


def predict_next_word(text):
    """Predict the next likely word using the saved LSTM model."""
    cleaned_text = " ".join(str(text).strip().split()).lower()
    if not cleaned_text:
        return ""

    sequence = tokenizer.texts_to_sequences([cleaned_text])[0]
    if not sequence:
        return ""

    padded_sequence = pad_sequences([sequence], maxlen=max_len, padding="pre")
    probabilities = model.predict(padded_sequence, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    for candidate_index in (predicted_index, predicted_index + 1):
        word = tokenizer.index_word.get(candidate_index)
        if word:
            return word

    return ""


st.set_page_config(page_title="Next Word Prediction", layout="centered")
st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word**.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        if next_word:
            st.success(f"**Predicted Next Word:** {next_word}")
        else:
            st.warning("The model could not find a prediction for this input.")

st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")