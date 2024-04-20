import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import CuDNNLSTM
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
import pickle
from sklearn.metrics import precision_score, recall_score
import json

from sklearn.metrics import f1_score

import Preprocessing as prep
import Tokenisers as tokenise


# Read the dataset
dataset = pd.read_json("Final_Model/Twitter/data_without_hashtags.json")

#########################################################################
# Preprocessing
x_name =  'text'
y_name = 'isSarcastic'

dataset = prep.oversample(dataset.text, dataset.isSarcastic, x_name, y_name)
dataset = prep.replaceEmoji_emoticon(dataset)
dataset['text'] = dataset.text.apply(prep.replace_abbreviations)
dataset['text'] = dataset.text.apply(prep.remove_user_mentions)

#########################################################################

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['isSarcastic'], test_size=0.3, random_state=42)


#########################################################################

# Tokenize
max_length = 140
X_train, X_test, tokenizer =tokenise.tweetTokenizer(X_train, X_test, max_length)

##################################################################

#Create model
embedding_dim = 150
vocab_size = len(tokenizer.word_index) + 1

optimizer = Adam(learning_rate=5.3254130613090156e-05)
m1 = Sequential()
m1.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
m1.add(LSTM(units=10))
m1.add(Dense(units=40, activation='relu'))
m1.add(Dense(units=20, activation='relu'))
m1.add(Dense(units=1, activation='sigmoid'))

m1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
m1.summary()

##########################################################################

# Training and getting results

m1.fit(X_train, y_train, epochs=12, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = m1.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy * 100:.2f}%')


# Predict on validation data
y_val_pred_prob_m1 = m1.predict(X_test)
y_val_pred_m1 = (y_val_pred_prob_m1 > 0.5).astype(int)  

# Calculate precision and recall for binary classification
precision_m1 = precision_score(y_test, y_val_pred_m1)
recall_m1 = recall_score(y_test, y_val_pred_m1)

# print the results
print(f'Precision: {precision_m1:.4f}')
print(f'Recall: {recall_m1:.4f}')

f1_m1 = f1_score(y_test, y_val_pred_m1)

print(f'F1 Score: {f1_m1:.2f}')


#####################################################################
# Saving

# Save the tokenizer
with open('Final_Model/Twitter/Tokenizer2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import joblib

#Save the model architecture
with open('Final_Model/Twitter/model_cudnn_lstm_architecture2.joblib', 'wb') as f:
    joblib.dump(m1.to_json(), f)

# Save the model weights
m1.save_weights('Final_Model/Twitter/model_cudnn_lstm_weights2.h5')

