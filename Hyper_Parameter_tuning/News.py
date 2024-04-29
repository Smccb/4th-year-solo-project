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

import Preprocessing as prep
import Tokenisers as tokenise
import text_mining_utils as tmu

dataset = pd.read_json("Datasets/Sarcasm_Headlines_Dataset.json", lines=True)

column_name_to_remove = 'article_link'
dataset = dataset.drop(columns=[column_name_to_remove])

#####################################################################
#preprocessing 

y_name = "is_sarcastic"
x_name = "headline"

dataset = prep.oversample(dataset.headline, dataset.is_sarcastic, x_name, y_name)
dataset['headline'] = dataset['headline'].str.lower()
dataset = prep.contractions_replaced(dataset , x_name)

#######################################################################
#split dataset into test and val set 70% 30%

X_train, X_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.3, random_state=42)

#######################################################################

#tokenizer
max_length = 100
X_train, X_test, tokenizer =tokenise.regTokeniser(X_train, X_test, max_length)

######################################################################
#saving
with open('Final_Model/Twitter/TokenizerN.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


test_data = [{'headline': text.tolist() if isinstance(text, np.ndarray) else text, 
              'is_sarcastic': int(label)} for text, label in zip(X_test, y_test)]

#####################################################################

#create model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    vocab_size = len(tokenizer.word_index) + 1
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=hp.Int('embedding_dim', min_value=50, max_value=150, step=25),
                        input_length=max_length))
    model.add(LSTM(units=hp.Int('lstm_units', min_value=10, max_value=100, step=10)))
    model.add(Dense(units=1, activation='sigmoid'))

    #define the learning rate as a tunable hyperparameter
    learning_rate = hp.Float('learning_rate', min_value=0.0000001, max_value=0.001, sampling='log')
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model



#########################################################################

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='hyperband',
    project_name='text_classification'
)


tuner.search(x=X_train, y=y_train, epochs=30, validation_data=(X_test, y_test))

#get the best model 
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()


##########################################################################

loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')