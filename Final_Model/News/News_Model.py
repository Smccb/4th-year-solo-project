import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import CuDNNLSTM
from keras.models import Sequential
from keras.layers import Embedding, Dense
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer
import pickle
import joblib
import json
import seaborn as sns

import Tokenisers as tokenise
import Preprocessing as prep
import text_mining_utils as tmu


dataset = pd.read_json("Datasets/Sarcasm_Headlines_Dataset.json", lines=True)

column_name_to_remove = 'article_link'
dataset = dataset.drop(columns=[column_name_to_remove])

##########################################################################
#Preproccessing
y_name = "is_sarcastic"
x_name = "headline"
# Random oversample
dataset = prep.oversample(dataset.headline, dataset.is_sarcastic, x_name, y_name)

# Lowercasing text
dataset['headline'] = dataset['headline'].str.lower()

# Replace contractions
dataset = prep.contractions_replaced(dataset , x_name)



##########################################################################
# Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.3, random_state=42)


column_names = ['toRemove', 'isSarcastic', 'text']
file_path = 'Combined/Test_v1.txt'
dataset = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

test_text = dataset['text']
test_labels = dataset['isSarcastic']

# Shuffle and quarter dataset
X_half1, X_half2, y_half1, y_half2 = train_test_split(test_text, test_labels, test_size=0.5, random_state=1)
X_test, X_quarter2, y_test, y_quarter2 = train_test_split(X_half1, y_half1, test_size=0.5, random_state=1)

#######################################################################

# Tokenizer
max_length = 100 #140 for using combined model saved otherwise 100
X_train, X_test, tokenizer =tokenise.regTokeniser(X_train, X_test, max_length)

######################################################################

#create model
embedding_dim = 175

#define the vocabulary size based on the actual number of unique words in the training data
vocab_size = len(tokenizer.word_index) + 1

optimizer = Adam(learning_rate=0.00006784)
m1 = Sequential()
m1.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
m1.add(LSTM(units=60))
m1.add(Dense(units=100,activation='relu'))
m1.add(Dense(units=20,activation='relu'))
m1.add(Dense(units=50,activation='relu'))
m1.add(Dense(units=1, activation='sigmoid'))

m1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
m1.summary()

########################################################################


#training and results
history = m1.fit(X_train, y_train, epochs=8, batch_size=128, validation_data=(X_test, y_test))

loss, accuracy = m1.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy * 100:.2f}%')

#predict on val data
y_val_pred_prob_m1 = m1.predict(X_test)
y_val_pred_m1 = (y_val_pred_prob_m1 > 0.5).astype(int)  

y_val_true_m1 = y_test

precision_m1 = precision_score(y_val_true_m1, y_val_pred_m1)
recall_m1 = recall_score(y_val_true_m1, y_val_pred_m1)

f1_m1 = f1_score(y_val_true_m1, y_val_pred_m1)


print(f'Precision: {precision_m1:.2f}')
print(f'Recall: {recall_m1:.2f}')
print(f'F1 Score: {f1_m1:.2f}')





#####################################################################
# Saving

#save the tokenizer
with open('Final_Model/News/TokenizerN.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import joblib

#save the model architecture
with open('Final_Model/News/model_cudnn_lstm_architectureN.joblib', 'wb') as f:
    joblib.dump(m1.to_json(), f)

#save the model weights
m1.save_weights('Final_Model/News/model_cudnn_lstm_weightsN.h5')

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
