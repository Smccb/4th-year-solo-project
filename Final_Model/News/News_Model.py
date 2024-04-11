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

import Tokenisers as tokenise
import Preprocessing as prep
import text_mining_utils as tmu


dataset = pd.read_json("Datasets/Sarcasm_Headlines_Dataset.json", lines=True)

column_name_to_remove = 'article_link'
dataset = dataset.drop(columns=[column_name_to_remove])

##########################################################################
#Preproccessing

#remove stopwords
#dataset['headline'] = dataset.headline.apply(prep.remove_stopwords) #worse results

#random oversample

#undersampleing
colName = "is_sarcastic"
dataset = prep.undersample(dataset, colName)

#replace abbreviations
#dataset['headline'] = dataset.headline.apply(prep.replace_abbreviations)

#lowercasing text
dataset['headline'] = dataset['headline'].str.lower()


#contradictions fixes
# contractions_dict = {
#     "ain't": "am not",
#     "aren't": "are not",
#     "can't": "cannot",
#     "could've": "could have",
#     "couldn't": "could not",
#     "didn't": "did not",
#     "doesn't": "does not",
#     "don't": "do not",
#     "hadn't": "had not",
#     "hasn't": "has not",
#     "haven't": "have not",
#     "he'd": "he would",
#     "he'll": "he will",
#     "he's": "he is",
#     "how'd": "how did",
#     "how'll": "how will",
#     "how's": "how is",
#     "i'd": "i would",
#     "i'll": "i will",
#     "i'm": "i am",
#     "i've": "i have",
#     "isn't": "is not",
#     "it'd": "it would",
#     "it'll": "it will",
#     "it's": "it is",
#     "let's": "let us",
#     "might've": "might have",
#     "must've": "must have",
#     "shan't": "shall not",
#     "she'd": "she would",
#     "she'll": "she will",
#     "she's": "she is",
#     "should've": "should have",
#     "shouldn't": "should not",
#     "that'll": "that will",
#     "that's": "that is",
#     "there's": "there is",
#     "they'd": "they would",
#     "they'll": "they will",
#     "they're": "they are",
#     "they've": "they have",
#     "wasn't": "was not",
#     "we'd": "we would",
#     "we'll": "we will",
#     "we're": "we are",
#     "we've": "we have",
#     "weren't": "were not",
#     "what'll": "what will",
#     "what're": "what are",
#     "what's": "what is",
#     "what've": "what have",
#     "when's": "when is",
#     "where'd": "where did",
#     "where's": "where is",
#     "where've": "where have",
#     "who'd": "who would",
#     "who'll": "who will",
#     "who's": "who is",
#     "who've": "who have",
#     "why'd": "why did",
#     "why'll": "why will",
#     "why's": "why is",
#     "won't": "will not",
#     "would've": "would have",
#     "wouldn't": "would not",
#     "you'd": "you would",
#     "you'll": "you will",
#     "you're": "you are",
#     "you've": "you have"
# }

#dataset['headline'] = dataset['headline'].apply(lambda x: tmu.resolve_contractions(x, contractions_dict))

#stemming
dataset.headline = dataset.headline.apply(tmu.stem_doc, stemmer=PorterStemmer())

#remove digits
#dataset['headline'] = dataset['headline'].apply(lambda x: tmu.remove_d(x))

##########################################################################

X_train, X_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.3, random_state=42)

#######################################################################

# Tokenizer
max_length = 140
X_train, X_test, tokenizer =tokenise.regTokeniser(X_train, X_test, max_length)

######################################################################

# Create model
embedding_dim = 100

# Define the vocabulary size based on the actual number of unique words in the training data
vocab_size = len(tokenizer.word_index) + 1

optimizer = Adam(learning_rate=0.000009)
m1 = Sequential()
m1.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
m1.add(CuDNNLSTM(units=150))
m1.add(Dense(units=64))
m1.add(Dense(units=64))
m1.add(Dense(units=1, activation='sigmoid'))

m1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
m1.summary()

########################################################################

# Training and Results
m1.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

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


#####################################################################
# Saving

# Save the tokenizer
with open('Final_Model/News/TokenizerN.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import joblib

# Save the model architecture
with open('Final_Model/News/model_cudnn_lstm_architectureN.joblib', 'wb') as f:
    joblib.dump(m1.to_json(), f)

# Save the model weights
m1.save_weights('Final_Model/News/model_cudnn_lstm_weightsN.h5')