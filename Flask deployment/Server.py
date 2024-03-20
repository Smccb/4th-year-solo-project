#code from https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c

import numpy as np
from flask import Flask, request, jsonify
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

# Load the model architecture
with open('Testing_GPU_training/model_cudnn_lstm_architecture.joblib', 'rb') as f:
    model_json = joblib.load(f)

m1_loaded = model_from_json(model_json)

# Load the model weights
m1_loaded.load_weights('Testing_GPU_training/model_cudnn_lstm_weights.h5')


def predict_sarcasm(user_input, model, tokenizer, max_length, threshold=0.5):
    # Tokenize and preprocess user input
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=max_length)

    # Make prediction
    prediction_prob = model.predict(user_input_padded)
    predicted_label = 1 if prediction_prob[0, 0] > threshold else 0

    return predicted_label, prediction_prob


@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # prediction = model.predict([[np.array(data['exp'])]])
    # output = prediction[0]

    tokenizer = Tokenizer()
    max_length = 138
    user_input = str(data)

    # Call the predict_sarcasm function with a custom threshold (e.g., 0.3)
    predicted_label, predicted_prob = predict_sarcasm(user_input, m1_loaded, tokenizer, max_length, threshold=0.3)

    # Display the prediction
    if predicted_label == 1:
        output = "The model predicts that the input is sarcastic."
    else:
        output = "The model predicts that the input is not sarcastic."

    # Optional: Display the predicted probabilities
    #print("Predicted Probabilities:", predicted_prob)
    #print("User input", user_input)


    #jsonify(output)
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

