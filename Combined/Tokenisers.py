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

#Regular tokeniser 
def regTokeniser(X_train, X_test, max_length):
    tokenizer = Tokenizer() # lower = False
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_length)

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_length)

    return X_train, X_test, tokenizer

####################################################################
# Tweet tokeniser 
def tweetTokenizer(X_train1, X_test1, X_train2, X_test2, max_length):

    tweetTokenizer = TweetTokenizer()
    X_train_tokenized1 = [tweetTokenizer.tokenize(text) for text in X_train1]
    X_test_tokenized1 = [tweetTokenizer.tokenize(text) for text in X_test1]
    X_train_tokenized2 = [tweetTokenizer.tokenize(text) for text in X_train2]
    X_test_tokenized2 = [tweetTokenizer.tokenize(text) for text in X_test2]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_tokenized1)
    tokenizer.fit_on_texts(X_train_tokenized2)

    X_train_sequences1 = tokenizer.texts_to_sequences(X_train_tokenized1)
    X_test_sequences1 = tokenizer.texts_to_sequences(X_test_tokenized1)
    X_train_sequences2 = tokenizer.texts_to_sequences(X_train_tokenized2)
    X_test_sequences2 = tokenizer.texts_to_sequences(X_test_tokenized2)

    # Pad sequences
    X_train1 = pad_sequences(X_train_sequences1, maxlen=max_length)
    X_test1 = pad_sequences(X_test_sequences1, maxlen=max_length)
    X_train2 = pad_sequences(X_train_sequences2, maxlen=max_length)
    X_test2 = pad_sequences(X_test_sequences2, maxlen=max_length)

    return X_train1, X_test1, X_train2, X_test2, tokenizer