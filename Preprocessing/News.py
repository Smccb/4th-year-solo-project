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

import Preprocessing as prep
import Tokenisers as tokenise
import text_mining_utils as tmu

dataset = pd.read_json("Datasets/Sarcasm_Headlines_Dataset.json", lines=True)

column_name_to_remove = 'article_link'
dataset = dataset.drop(columns=[column_name_to_remove])

#####################################################################
# Preprocessing 
y_name = "is_sarcastic"
x_name = "headline"

# Random oversample
dataset = prep.oversample(dataset.headline, dataset.is_sarcastic, x_name, y_name)
# Lowercasing text
dataset['headline'] = dataset['headline'].str.lower()
# Replace contractions
dataset = prep.contractions_replaced(dataset , x_name)


#######################################################################

X_train, X_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.3, random_state=42)

#######################################################################

# Tokenizer
max_length = 100
X_train, X_test, tokenizer =tokenise.regTokeniser(X_train, X_test, max_length)

######################################################################


# Create model
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1
optimizer = Adam(learning_rate=0.0001)

m1 = Sequential()
m1.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
# m1.add(CuDNNLSTM(units=10))
# m1.add(Dense(units=1, activation='sigmoid'))

m1.add(LSTM(units=10))
m1.add(Dense(units=1, activation='sigmoid'))

m1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
m1.summary()

##########################################################################

# Training and getting results

history = m1.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

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

#-------------------------------------------

# import pickle

# # Save the tokenizer
# with open('non_GPU_model_Tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Load the pickled model
# m1.save('non_GPU_model_trained.h5')

import matplotlib.pyplot as plt

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')  # Adjust if different key
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Adjust if different key
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()


from sklearn.metrics import f1_score
# Data to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision_m1, recall_m1, f1_m1]

# # Creating the bar plot
# plt.figure(figsize=(10, 5))
# sns.barplot(metrics, values)
# plt.title('Model Performance Metrics')
# plt.ylabel('Value')
# for i, value in enumerate(values):
#     plt.text(i, value, f'{value:.2f}', ha = 'center', va = 'bottom')
# plt.show()