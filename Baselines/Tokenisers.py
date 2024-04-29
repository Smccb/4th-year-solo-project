#This file has the tokenisers for this project

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
from nltk.tokenize import TweetTokenizer

#regular tokeniser 
def regTokeniser(X_train, X_test, max_length):
    tokenizer = Tokenizer() # lower = False #can make it so tokenizer does not lowercase text
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_length)

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_length)

    return X_train, X_test, tokenizer

####################################################################
#tweet tokeniser 
def tweetTokenizer(X_train, X_test, max_length):

    tweetTokenizer = TweetTokenizer()
    X_train_tokenized = [tweetTokenizer.tokenize(text) for text in X_train]
    X_test_tokenized = [tweetTokenizer.tokenize(text) for text in X_test]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_tokenized)

    X_train_sequences = tokenizer.texts_to_sequences(X_train_tokenized)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_tokenized)

    #pad sequences
    X_train = pad_sequences(X_train_sequences, maxlen=max_length)
    X_test = pad_sequences(X_test_sequences, maxlen=max_length)

    return X_train, X_test, tokenizer