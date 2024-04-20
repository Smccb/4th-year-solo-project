import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))



import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Function definition (assuming you have the function already defined)
def tweetTokenizer(X_train, X_test, max_length):
    tweet_tokenizer = TweetTokenizer()
    X_train_tokenized = [tweet_tokenizer.tokenize(text) for text in X_train]
    X_test_tokenized = [tweet_tokenizer.tokenize(text) for text in X_test]

    print("Tokenized Training Data:")
    print(X_train_tokenized)
    print("\nTokenized Test Data:")
    print(X_test_tokenized)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_tokenized)

    X_train_sequences = tokenizer.texts_to_sequences(X_train_tokenized)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_tokenized)

    print("\nTraining Sequences:")
    print(X_train_sequences)
    print("\nTest Sequences:")
    print(X_test_sequences)

    # Pad sequences
    X_train = pad_sequences(X_train_sequences, maxlen=max_length)
    X_test = pad_sequences(X_test_sequences, maxlen=max_length)

    return X_train, X_test, tokenizer

# Create a sample dataset
data = {
    'tweets': [
        "Hello world! #sunny",
        "Looking forward to the weekend :)",
        "What a game last night! #exciting",
        "Can't believe it's already Thursday...",
        "Let's go for a run 🏃",
        "Good night all! #tired",
        "Loving the new playlist!! <3",
        "Happy birthday to me! 🎂",
        "That's interesting... #thoughts",
        "Feeling blessed ❤️"
    ]
}

df = pd.DataFrame(data)

# Split the dataset
X_train, X_test = train_test_split(df['tweets'], test_size=0.2, random_state=42)

# Use the tokenizer function
max_length = 10  # Define the maximum length of the sequences
X_train_prepared, X_test_prepared, tokenizer = tweetTokenizer(X_train, X_test, max_length)

# Additional display of how tokens map to words
print("\nWord Index (Token Mapping):")
print(tokenizer.word_index)

