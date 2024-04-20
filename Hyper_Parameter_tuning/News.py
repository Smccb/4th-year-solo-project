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

# Remove stopwords
#dataset['headline'] = dataset.headline.apply(prep.remove_stopwords) #worse results
#dataset['headline'] = dataset['headline'].apply(prep.remove_punctuation )

y_name = "is_sarcastic"
x_name = "headline"
# Random oversample
dataset = prep.oversample(dataset.headline, dataset.is_sarcastic, x_name, y_name)
# Undersampleing
#dataset = prep.undersample(dataset, y_name)

# Replace abbreviations
#dataset['headline'] = dataset.headline.apply(prep.replace_abbreviations)

# Lowercasing text
dataset['headline'] = dataset['headline'].str.lower()

# Replace contractions
dataset = prep.contractions_replaced(dataset , x_name)

# Stemming
#dataset.headline = dataset.headline.apply(tmu.stem_doc, stemmer=PorterStemmer())

#######################################################################

X_train, X_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.3, random_state=42)

#######################################################################

# Tokenizer
max_length = 100
X_train, X_test, tokenizer =tokenise.regTokeniser(X_train, X_test, max_length)

######################################################################

import pickle

with open('Final_Model/Twitter/TokenizerN.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import json

test_data = [{'headline': text.tolist() if isinstance(text, np.ndarray) else text, 
              'is_sarcastic': int(label)} for text, label in zip(X_test, y_test)]

# Save test_data to a JSON file
with open('test_dataN.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

# Do the same for training data
train_data = [{'headline': text.tolist() if isinstance(text, np.ndarray) else text, 
               'is_sarcastic': int(label)} for text, label in zip(X_train, y_train)]

# Save train_data to a JSON file
with open('train_dataN.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)


######################################################################


# # Create model
# from keras.models import Sequential
# from keras.layers import Embedding, LSTM, Dense
# from keras.optimizers import Adam
# from keras_tuner import Hyperband
# import keras_tuner as kt

# def build_model(hp):
#     model = Sequential()
#     vocab_size = len(tokenizer.word_index) + 1

#     model.add(Embedding(input_dim=vocab_size, output_dim=hp.Int('embedding_dim', min_value=50, max_value=200, step=25), input_length=max_length))
#     model.add(LSTM(units=hp.Int('lstm_units', min_value=10, max_value=250, step=10)))

#     for i in range(hp.Int('num_dense_layers', 1, 5)):
#         model.add(Dense(units=hp.Int('dense_units_' + str(i), min_value=10, max_value=100, step=10), activation='relu'))

#     model.add(Dense(units=1, activation='sigmoid'))

#     learning_rate = hp.Float('learning_rate', min_value=0.0000001, max_value=0.0001, sampling='log')
#     optimizer = Adam(learning_rate=learning_rate)

#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# tuner = kt.RandomSearch(
#     hypermodel=build_model,
#     objective="val_accuracy",
#     max_trials=20,
#     executions_per_trial=2,
#     overwrite=True,
#     directory='kt_random_search_news',
#     project_name='classification'
# )


# tuner.search(x=X_train, y=y_train,
#              epochs=100,
#              validation_data=(X_test, y_test),
#              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)])

# # Get the best model
# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.summary()


# ##########################################################################

# loss, accuracy = best_model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')



# Saving

#with open('Final_Model/News/TokenizerN.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# import json

# test_data = [{'text': text.tolist() if isinstance(text, np.ndarray) else text, 
#               'isSarcastic': int(label)} for text, label in zip(X_test, y_test)]

# # Save test_data to a JSON file
# with open('test_data.json', 'w', encoding='utf-8') as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=4)

# # Do the same for training data
# train_data = [{'text': text.tolist() if isinstance(text, np.ndarray) else text, 
#                'isSarcastic': int(label)} for text, label in zip(X_train, y_train)]

# # Save train_data to a JSON file
# with open('train_data.json', 'w', encoding='utf-8') as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=4)



##################################################################

# # Create model
# from keras.models import Sequential
# from keras.layers import Embedding, LSTM, Dense
# from keras.optimizers import Adam
# from keras_tuner import Hyperband

# def build_model(hp):
#     model = Sequential()
#     vocab_size = len(tokenizer.word_index) + 1

#     model.add(Embedding(input_dim=vocab_size, output_dim=hp.Int('embedding_dim', min_value=50, max_value=200, step=25), input_length=max_length))
#     model.add(LSTM(units=hp.Int('lstm_units', min_value=10, max_value=250, step=10)))

#     for i in range(hp.Int('num_dense_layers', 1, 5)):
#         model.add(Dense(units=hp.Int('dense_units_' + str(i), min_value=10, max_value=100, step=10), activation='relu'))

#     model.add(Dense(units=1, activation='sigmoid'))

#     learning_rate = hp.Float('learning_rate', min_value=0.0000001, max_value=0.0001, sampling='log')
#     optimizer = Adam(learning_rate=learning_rate)

#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model
# # Instantiate the tuner
# tuner = Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=150,
#     factor=10,
#     directory='hyperband_news',
#     project_name='classification'
# )


# tuner.search(x=X_train, y=y_train, epochs=150, validation_data=(X_test, y_test))

# # Get the best model
# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.summary()


# ##########################################################################

# loss, accuracy = best_model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Create model
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

    # Define the learning rate as a tunable hyperparameter
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

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()


##########################################################################

loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')