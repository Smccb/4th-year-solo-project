from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd

from keras.models import model_from_json
import joblib

app = Flask(__name__)

max_length = 140

with open('Flask\Flask_GPU\model_cudnn_lstm_architecture2.joblib', 'rb') as f:
    model_json = joblib.load(f)

m1_loaded = model_from_json(model_json)
m1_loaded.load_weights('Flask\Flask_GPU\model_cudnn_lstm_weights2.h5')

import pickle

path = "Flask\Flask_GPU\Tokenizer2.pickle"

# Load the tokenizer
with open(path, 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_sarcasm(user_input, model, tokenizer, max_length, threshold=0.5):
    # Tokenize and preprocess user input
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=max_length)

    # Make prediction
    prediction_prob = model.predict(user_input_padded)
    predicted_label = 1 if prediction_prob[0, 0] > threshold else 0

    return predicted_label, prediction_prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    predicted_label, predicted_prob = predict_sarcasm(user_input,m1_loaded,tokenizer, max_length, threshold=0.3)
    
    if predicted_label == 1:
        output = "The model predicts that the input is sarcastic."
    else:
        output = "The model predicts that the input is not sarcastic."
    return render_template('result.html', prediction=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

