import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import CuDNNLSTM
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import seaborn as sns

import Tokenisers as tokenise
import Preprocessing as prep


# Read the dataset
dataset = pd.read_json("Final_Model/Twitter/data_without_hashtags.json")

#########################################################################
#preprocessing
x_name =  'text'
y_name = 'isSarcastic'

dataset = prep.oversample(dataset.text, dataset.isSarcastic, x_name, y_name)
dataset = prep.replaceEmoji_emoticon(dataset)
dataset['text'] = dataset.text.apply(prep.replace_abbreviations)
dataset['text'] = dataset.text.apply(prep.remove_user_mentions)

#########################################################################

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['isSarcastic'], test_size=0.3, random_state=42)

#testing news data on twitter model
# file_path = 'Combined/Test_data1.json'
# dataset = pd.read_json(file_path)

# test_text = dataset['text']
# test_labels = dataset['isSarcastic']

# # Shuffle and quarter dataset
# X_half1, X_half2, y_half1, y_half2 = train_test_split(test_text, test_labels, test_size=0.5, random_state=1)
# X_test, X_quarter2, y_test, y_quarter2 = train_test_split(X_half1, y_half1, test_size=0.5, random_state=1)

#########################################################################

#tokenize data
max_length = 140 # this is the padding number
X_train, X_test, tokenizer =tokenise.tweetTokenizer(X_train, X_test, max_length)

##################################################################

#create model
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

#training and getting results
history = m1.fit(X_train, y_train, epochs=12, batch_size=128, validation_data=(X_test, y_test))

#evaluate the model
loss, accuracy = m1.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy * 100:.2f}%')


#predict on validation data
y_val_pred_prob_m1 = m1.predict(X_test)
y_val_pred_m1 = (y_val_pred_prob_m1 > 0.5).astype(int)  

#calculate precision and recall for binary classification
precision_m1 = precision_score(y_test, y_val_pred_m1)
recall_m1 = recall_score(y_test, y_val_pred_m1)

#print the results
print(f'Precision: {precision_m1:.4f}')
print(f'Recall: {recall_m1:.4f}')

f1_m1 = f1_score(y_test, y_val_pred_m1)

print(f'F1 Score: {f1_m1:.2f}')

#####################################################################

import matplotlib.pyplot as plt

#plotting train and val loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

#plotting train and val accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

#data to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision_m1, recall_m1, f1_m1]


#####################################################################
# Saving
import joblib

#save the tokenizer
with open('Final_Model/Twitter/Tokenizer2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#save the model architecture
with open('Final_Model/Twitter/model_cudnn_lstm_architecture2.joblib', 'wb') as f:
    joblib.dump(m1.to_json(), f)

#save the model weights
m1.save_weights('Final_Model/Twitter/model_cudnn_lstm_weights2.h5')

