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
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
import pickle
from sklearn.metrics import precision_score, recall_score
import json
import Preprocessing as prep
import Tokenisers as tokenise

import text_mining_utils as tmu

from sklearn.metrics import f1_score


# Read the dataset
dataset1 = pd.read_json("Final_Model/Twitter/data_without_hashtags.json")
dataset2 = pd.read_json("Datasets/Sarcasm_Headlines_Dataset.json", lines=True)

#########################################################################
#preprocessing
# Dataset 1 (Twitter)
x_name1 =  'text'
y_name1 = 'isSarcastic'

dataset1 = prep.oversample(dataset1.text, dataset1.isSarcastic, x_name1, y_name1)
dataset1 = prep.replaceEmoji_emoticon(dataset1)
dataset1['text'] = dataset1.text.apply(prep.replace_abbreviations)
dataset1['text'] = dataset1.text.apply(prep.remove_user_mentions)

# Dataset 2 (News)
y_name2 = "is_sarcastic"
x_name2 = "headline"

dataset2 = prep.oversample(dataset2.headline, dataset2.is_sarcastic, x_name2, y_name2)
dataset2['headline'] = dataset2['headline'].str.lower()
dataset2 = prep.contractions_replaced(dataset2 , x_name2)

#########################################################################

#this was code for testing
# file_path = 'Final_Model/Twitter/Test_data.json'

# temp_df = pd.read_json(file_path)
# test_text = temp_df['text']
# test_labels = temp_df['isSarcastic']

#split the dataset into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset1['text'], dataset1['isSarcastic'], test_size=0.3, random_state=42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(dataset2['headline'], dataset2['is_sarcastic'], test_size=0.3, random_state=42)


#testing for news dataset test
# data = [{"headline": text, "is_sarcastic": label} for text, label in zip(X_test, y_test)]
# # Writing the data to a JSON file
# with open('test_data.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4)

# print("Data has been written to test_data.json")

#X_test = test_text
#y_test = test_labels

#########################################################################

#tokenize data
max_length = 140
X_train1, X_test1, X_train2, X_test2, tokenizer =tokenise.tweetTokenizer(X_train1, X_test1, X_train2, X_test2, max_length)

##################################################################

#create model
embedding_dim = 175
vocab_size = len(tokenizer.word_index) + 1

optimizer = Adam(learning_rate=5.3254130613090156e-05) #learning rate in scientific notation
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

# News
history1 = m1.fit(
    X_train2, y_train2,
    batch_size=128,
    epochs=10,
    validation_data=(X_test2, y_test2)
)

# Twitter
history2 = m1.fit(
    X_train1, y_train1,
    batch_size=128,
    epochs=10,
    validation_data=(X_test1, y_test1)
)



loss1, accuracy1 = m1.evaluate(X_test1, y_test1)
loss2, accuracy2 = m1.evaluate(X_test2, y_test2)
print(f'Test Accuracy for Dataset Twitter: {accuracy1*100:.2f}%')
print(f'Test Accuracy for Dataset News: {accuracy2*100:.2f}%')



#predict on validation data
y_val_pred_prob_m1 = m1.predict(X_test1)
y_val_pred_m1 = (y_val_pred_prob_m1 > 0.5).astype(int)  

#calculate precision and recall for binary classification
precision_m1 = precision_score(y_test1, y_val_pred_m1)
recall_m1 = recall_score(y_test1, y_val_pred_m1)

#print the results
print(f'Precision1: {precision_m1:.4f}')
print(f'Recall1: {recall_m1:.4f}')
f1_m1 = f1_score(y_test1, y_val_pred_m1)
print(f'F1 Score1: {f1_m1:.2f}')

#predict on validation data
y_val_pred_prob_m1 = m1.predict(X_test2)
y_val_pred_m1 = (y_val_pred_prob_m1 > 0.5).astype(int)  

#calculate precision and recall for binary classification
precision_m1 = precision_score(y_test2, y_val_pred_m1)
recall_m1 = recall_score(y_test2, y_val_pred_m1)

# print the results
print(f'Precision2: {precision_m1:.4f}')
print(f'Recall2: {recall_m1:.4f}')
f1_m1 = f1_score(y_test2, y_val_pred_m1)
print(f'F1 Score2: {f1_m1:.2f}')

#####################################################################
#saving
import joblib


# save tokenizer
with open('Combined/Tokenizer_combined.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save model architecture
with open('Combined/model_cudnn_lstm_architectureC.joblib', 'wb') as f:
    joblib.dump(m1.to_json(), f)

# Save the model weights
m1.save_weights('Combined/model_cudnn_lstm_weightsC.h5')

