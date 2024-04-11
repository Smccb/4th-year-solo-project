import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
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
from sklearn.metrics import f1_score

import Preprocessing as prep
import Tokenisers as tokenise


# Read the dataset
dataset = pd.read_json("Final_Model/Twitter/data_without_hashtags.json")

#####################################################################
#Preprocessing 



#######################################################################

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['isSarcastic'], test_size=0.3, random_state=42)

#########################################################################

# Tokenize
max_length = 140
X_train, X_test, tokenizer =tokenise.regTokeniser(X_train, X_test, max_length)

##################################################################

# Create model
embedding_dim = 50  
vocab_size = 1000

m1 = Sequential()
m1.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
m1.add(CuDNNLSTM(units=10))
m1.add(Dense(units=1, activation='sigmoid'))

m1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m1.summary()

##########################################################################

# Training and getting results

m1.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = m1.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy * 100:.2f}%')

from sklearn.metrics import precision_score, recall_score

# Predict on validation data
y_val_pred_prob_m1 = m1.predict(X_test)
y_val_pred_m1 = (y_val_pred_prob_m1 > 0.5).astype(int)  

y_val_true_m1 = y_test

# Calculate precision and recall for binary classification
precision_m1 = precision_score(y_val_true_m1, y_val_pred_m1)
recall_m1 = recall_score(y_val_true_m1, y_val_pred_m1)

# print the results
print(f'Precision: {precision_m1:.4f}')
print(f'Recall: {recall_m1:.4f}')

f1_m1 = f1_score(y_val_true_m1, y_val_pred_m1)

print(f'F1 Score: {f1_m1:.2f}')
