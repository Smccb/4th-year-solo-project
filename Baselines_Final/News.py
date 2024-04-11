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

import Tokenisers as tokenise


dataset = pd.read_json("Datasets/Sarcasm_Headlines_Dataset.json", lines=True)

column_name_to_remove = 'article_link'
dataset = dataset.drop(columns=[column_name_to_remove])

X_train, X_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.3, random_state=42)

#######################################################################

# Tokenizer
max_length = 100
X_train, X_test, tokenizer =tokenise.regTokeniser(X_train, X_test, max_length)

######################################################################

# Create model
embedding_dim = 50  
vocab_size = 10000  
max_length = 100

m1 = Sequential()
m1.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
m1.add(CuDNNLSTM(units=10))
m1.add(Dense(units=1, activation='sigmoid'))

m1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m1.summary()

########################################################################

# Training and Results
m1.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

loss, accuracy = m1.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy * 100:.2f}%')

# Predict on validation data
y_val_pred_prob_m1 = m1.predict(X_test)
y_val_pred_m1 = (y_val_pred_prob_m1 > 0.5).astype(int)  

y_val_true_m1 = y_test

precision_m1 = precision_score(y_val_true_m1, y_val_pred_m1)
recall_m1 = recall_score(y_val_true_m1, y_val_pred_m1)

f1_m1 = f1_score(y_val_true_m1, y_val_pred_m1)

print(f'F1 Score: {f1_m1:.2f}')

print(precision_m1)
print(recall_m1)


# 293/293 [==============================] - 5s 12ms/step - loss: 0.4915 - accuracy: 0.7484 - val_loss: 0.3516 - val_accuracy: 0.8502
# Epoch 2/10
# 293/293 [==============================] - 3s 11ms/step - loss: 0.2558 - accuracy: 0.9008 - val_loss: 0.3442 - val_accuracy: 0.8530
# Epoch 3/10
# 293/293 [==============================] - 3s 11ms/step - loss: 0.1672 - accuracy: 0.9408 - val_loss: 0.3702 - val_accuracy: 0.8521
# Epoch 4/10
# 293/293 [==============================] - 3s 10ms/step - loss: 0.1145 - accuracy: 0.9621 - val_loss: 0.4096 - val_accuracy: 0.8430
# Epoch 5/10
# 293/293 [==============================] - 3s 11ms/step - loss: 0.0759 - accuracy: 0.9777 - val_loss: 0.4807 - val_accuracy: 0.8445
# Epoch 6/10
# 293/293 [==============================] - 3s 10ms/step - loss: 0.0535 - accuracy: 0.9842 - val_loss: 0.5327 - val_accuracy: 0.8420
# Epoch 7/10
# 293/293 [==============================] - 3s 11ms/step - loss: 0.0373 - accuracy: 0.9891 - val_loss: 0.6233 - val_accuracy: 0.8391
# Epoch 8/10
# 293/293 [==============================] - 3s 11ms/step - loss: 0.0265 - accuracy: 0.9927 - val_loss: 0.6603 - val_accuracy: 0.8383
# Epoch 9/10
# 293/293 [==============================] - 3s 11ms/step - loss: 0.0213 - accuracy: 0.9938 - val_loss: 0.7066 - val_accuracy: 0.8396
# Epoch 10/10
# 293/293 [==============================] - 3s 11ms/step - loss: 0.0181 - accuracy: 0.9944 - val_loss: 0.7241 - val_accuracy: 0.8376
# 251/251 [==============================] - 1s 4ms/step - loss: 0.7241 - accuracy: 0.8376
# Loss: 0.724076509475708, Accuracy: 83.76%
# 251/251 [==============================] - 1s 2ms/step
# F1 Score: 0.82
# 0.8158774373259053 # precision
# 0.8206780610815354 #recall