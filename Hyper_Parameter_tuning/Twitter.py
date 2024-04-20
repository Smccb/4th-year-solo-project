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
import seaborn as sns

import Preprocessing as prep
import Tokenisers as tokenise


# Read the dataset
dataset = pd.read_json("Hyper_Parameter_tuning/data_without_hashtags.json")

#####################################################################
#Preprocessing 
x_name =  'text'
y_name = 'isSarcastic'

dataset = prep.oversample(dataset.text, dataset.isSarcastic, x_name, y_name)
dataset = prep.replaceEmoji_emoticon(dataset)
dataset['text'] = dataset.text.apply(prep.replace_abbreviations)
dataset['text'] = dataset.text.apply(prep.remove_user_mentions)


#######################################################################

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['isSarcastic'], test_size=0.3, random_state=42)

#########################################################################

# Tokenizer

max_length = 140
X_train, X_test, tokenizer =tokenise.tweetTokenizer(X_train, X_test, max_length)

########################################################################

# Saving

with open('Final_Model/Twitter/TokenizerT2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

# Create model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from keras_tuner import Hyperband

def build_model(hp):
    model = Sequential()
    vocab_size = len(tokenizer.word_index) + 1

    model.add(Embedding(input_dim=vocab_size, output_dim=hp.Int('embedding_dim', min_value=50, max_value=200, step=25), input_length=max_length))
    model.add(CuDNNLSTM(units=hp.Int('lstm_units', min_value=10, max_value=250, step=10)))

    for i in range(hp.Int('num_dense_layers', 1, 5)):
        model.add(Dense(units=hp.Int('dense_units_' + str(i), min_value=10, max_value=100, step=10), activation='relu'))

    model.add(Dense(units=1, activation='sigmoid'))

    learning_rate = hp.Float('learning_rate', min_value=0.0000001, max_value=0.0001, sampling='log')
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Instantiate the tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=150,
    factor=10,
    directory='hyperband2',
    project_name='classification'
)


tuner.search(x=X_train, y=y_train, epochs=150, validation_data=(X_test, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()


##########################################################################

loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')








