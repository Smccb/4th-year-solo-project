from keras_tuner import Hyperband
import tensorflow as tf
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
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
import keras_tuner as kt
import json
import os

import Preprocessing as prep
import Tokenisers as tokenise


with open('Final_Model/Twitter/TokenizerN.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 140

def build_model(hp):
    model = Sequential()
    vocab_size = len(tokenizer.word_index) + 1

    model.add(Embedding(input_dim=vocab_size, output_dim=hp.Int('embedding_dim', min_value=50, max_value=200, step=25), input_length=max_length))
    model.add(LSTM(units=hp.Int('lstm_units', min_value=10, max_value=250, step=10)))

    for i in range(hp.Int('num_dense_layers', 1, 5)):
        model.add(Dense(units=hp.Int('dense_units_' + str(i), min_value=10, max_value=100, step=10), activation='relu'))

    model.add(Dense(units=1, activation='sigmoid'))

    learning_rate = hp.Float('learning_rate', min_value=0.0000001, max_value=0.0001, sampling='log')
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=2,
    overwrite=True,
    directory='kt_random_search_news',
    project_name='classification'
)

# Reload the tuner from the saved state
tuner.reload()

# # Retrieve the best model without re-running the search
# models = tuner.get_best_models(num_models=1)


# for model in models:
#     model.summary()

# Retrieve the best hyperparameters
#best_hyperparameters = tuner.get_best_hyperparameters()[0]  # Corrected method call
#print("Best hyperparameters: ", best_hyperparameters.values)

# # Optionally, access all trials to analyze their outcomes
# all_trials = tuner.oracle.get_trial()
# for trial in all_trials:
#     print(trial.trial_id, trial.hyperparameters.values, trial.score)


# import os
# import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Directory where the trials are stored
dir = "kt_random_search_news"
next_dir = "classification"
base_dir = os.path.join(dir, next_dir)

#print(base_dir)


try:
    trial_directories = [f for f in os.listdir(os.path.join(dir, next_dir)) if f.startswith('trial_')]
    print("Found trial directories:", trial_directories)
except FileNotFoundError:
    print(f"Directory not found: {base_dir}")
except PermissionError:
    print(f"Permission denied: {base_dir}")

# Iterate through each trial directory
all_trials_data = []
for trial_folder in sorted(os.listdir(os.path.join(dir, next_dir))):
    if trial_folder.startswith('trial_'):
        trial_path = os.path.join(dir, next_dir, trial_folder, 'trial.json')
        if os.path.exists(trial_path):
            trial_data = load_json(trial_path)
            if trial_data['status'] == 'COMPLETED':
                # Collect data only from completed trials
                all_trials_data.append(trial_data)
                print(f"Trial ID: {trial_data['trial_id']}, Score: {trial_data.get('score')}, Hyperparameters: {trial_data['hyperparameters']['values']}")
            else:
                # Log failed trials for review
                print(f"Trial ID: {trial_data['trial_id']} failed with message: {trial_data['message']}")

# Find the best trial if scores are available and valid
if all_trials_data:
    best_trial = max((trial for trial in all_trials_data if trial.get('score') is not None), key=lambda x: x['score'], default=None)
    if best_trial:
        print("\nBest Trial:")
        print(f"Trial ID: {best_trial['trial_id']}")
        print(f"Score: {best_trial['score']}")
        print("Hyperparameters:", best_trial['hyperparameters']['values'])
else:
    print("No successful trials found.")


