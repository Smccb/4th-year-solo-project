import pickle
from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import joblib
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score
import Preprocessing as prep

# Load from saved/trainedl models

# Load the Twitter model and tokenizer
with open('Combined/model_cudnn_lstm_architecture2.joblib', 'rb') as f:
    m1_twitter = model_from_json(joblib.load(f))

m1_twitter.load_weights('Combined/model_cudnn_lstm_weights2.h5')

with open('Combined/Tokenizer2.pickle', 'rb') as handle:
    tokenizer_twitter = pickle.load(handle)

# Load the News model and tokenizer
with open('Combined/model_cudnn_lstm_architectureN.joblib', 'rb') as f:
    m1_news = model_from_json(joblib.load(f))

m1_news.load_weights('Combined/model_cudnn_lstm_weightsN.h5')

with open('Combined/TokenizerN.pickle', 'rb') as handle:
    tokenizer_news = pickle.load(handle)

############################################################

# Max length for padding
max_length = 140

# Updated predict sarcam function
def predict_sarcasm_ensemble(user_input, models, tokenizers, max_length, threshold= 0.5):
    predictions = []
    for model, tokenizer in zip(models, tokenizers):
        # Tokenize and preprocess user input
        user_input_sequence = tokenizer.texts_to_sequences([user_input])
        user_input_padded = pad_sequences(user_input_sequence, maxlen=max_length)

        # Make prediction
        prediction_prob = model.predict(user_input_padded)
        predictions.append(prediction_prob[0, 0])
    
    # Average predictions from models
    avg_prediction = sum(predictions) / 2
    predicted_label = 1 if avg_prediction > threshold else 0

    return predicted_label, avg_prediction

###################################################################

# # Take User input
# user_input = input("Enter a sentence: ")
# predicted_label, avg_prediction = predict_sarcasm_ensemble(
#     user_input, 
#     [m1_twitter, m1_news], 
#     [tokenizer_twitter, tokenizer_news], 
#     max_length, 
#     threshold=0.5
# )

# if predicted_label == 1:
#     print("Input is sarcastic.")
# else:
#     print("Input is not sarcastic.")

# print("Average Predicted Probability:", avg_prediction)

##################################################################

# # Test model accuracy

#file_path = 'Combined/Test_data1.json'
#dataset = pd.read_json(file_path

column_names = ['toRemove', 'isSarcastic', 'text']
file_path = 'Combined/Test_v1.txt'
dataset = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

test_text = dataset['text']
test_labels = dataset['isSarcastic']

# Shuffle and quarter dataset
X_half1, X_half2, y_half1, y_half2 = train_test_split(test_text, test_labels, test_size=0.5, random_state=1)
X_quarter1, X_quarter2, y_quarter1, y_quarter2 = train_test_split(X_half1, y_half1, test_size=0.5, random_state=1)
X_quarter3, X_quarter4, y_quarter3, y_quarter4 = train_test_split(X_half2, y_half2, test_size=0.5, random_state=1)

predicted_labels = []
for input_text in X_quarter1:
    predicted_label, _ = predict_sarcasm_ensemble(input_text, [m1_twitter, m1_news], [tokenizer_twitter, tokenizer_news], max_length, threshold=0.5)
    predicted_labels.append(predicted_label)

accuracy = accuracy_score(y_quarter1, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')


precision_m1 = precision_score(y_quarter1, predicted_labels)
recall_m1 = recall_score(y_quarter1, predicted_labels)

# print the results
print(f'Precision: {precision_m1:.4f}')
print(f'Recall: {recall_m1:.4f}')
f1_m1 = f1_score(y_quarter1, predicted_labels)
print(f'F1 Score: {f1_m1:.2f}')