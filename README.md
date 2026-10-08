# Next Word Prediction App

This project uses a saved LSTM model and a tokenizer to predict the next word from a sentence.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- The app expects the model file named `lstm_model (1).h5` in the same folder.
- It also loads `tokenizer.pkl` and `max_len.pkl` from the project directory.
- The prediction logic pads sequences using the saved length from training to avoid shape mismatches.
