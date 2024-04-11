from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd

app = Flask(__name__)

max_length = 138

path = 'Sarcasm datasets\Twitter_2\model_trained.h5'

# Load the Keras model
m1_loaded = load_model(path)
import pickle

# Load the tokenizer
with open('Sarcasm datasets\Twitter_2\Tokenizer.pickle', 'rb') as handle:
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
    app.run(debug=True)
