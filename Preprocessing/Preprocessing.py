#All preprocessing functions

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import emoji
import json

################################################################################

def remove_Sarcasm_hashtag():
    file_path = 'Datasets/Train_v1.txt'
    column_names = ['toRemove', 'isSarcastic', 'text']

    # Read the dataset
    data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

    # Define patterns to remove
    patterns = [
        r'#sarcasm\b',
        r'#not\b',
        r'#Not\b',
        r'#sarcastic\b',
        r'#yeahright'

    ]

    # Remove patterns from text
    for pattern in patterns:
        data['text'] = data['text'].apply(lambda x: re.sub(pattern, '', x))

    # Drop the 'toRemove' column
    data.drop(columns=['toRemove'], inplace=True)

    # Convert DataFrame to dictionary
    data_dict = data.to_dict()

    # Save the dictionary to a JSON file
    with open('Final_Model/Twitter/cleaned_sarcasm.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

#remove_Sarcasm_hashtag()

#######################################################################################

# Undersampling
def undersample(dataset, colName):
    class_counts = dataset[colName].value_counts()

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    minority_class_count = class_counts[minority_class]

    majority_class_sampled = dataset[dataset[colName] == majority_class].sample(n=minority_class_count, random_state=42)

    balanced_data = pd.concat([majority_class_sampled, dataset[dataset[colName] == minority_class]])

    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=1).reset_index(drop=True)
    return balanced_data

################################################

# Remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

######################################################

# Emoji and emoticon replacement

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def replace_emoticons(text):
    emoticon_dict = {
        ':)': 'smile',
        ':(': 'frown',
        ':D': 'big smile',
        ':P': 'tongue out',
        ';)': 'wink',
        ':O': 'surprise',
        ':|': 'neutral',
        ':/': 'uncertain',
        ":'(": 'tears of sadness',
        ":'D": 'tears of joy',
        ':*': 'kiss',
        ':@': 'angry',
        ':x': 'mouth shut',
        ':3': 'cute',
        ':$': 'embarrassed',
        ":')": 'single tear',
        ':p': 'tongue out'
    }
    emoticon_dict_lower = {key.lower(): value for key, value in emoticon_dict.items()}
    
    pattern = re.compile(r"[:;]['-]?[)DPO|/\\@x3*$p]", re.IGNORECASE)

    # function to replace match with corresponding word
    def replace_func(match):
        return emoticon_dict_lower.get(match.group().lower(), match.group())

    return pattern.sub(replace_func, text)

def replaceEmoji_emoticon(dataset):
    dataset.text = dataset.text.apply(replace_emoticons)
    dataset.text = dataset.text.apply(replace_emojis)
    return dataset

######################################################################

# Remove @usersname
def remove_user_mentions(text):
    pattern = re.compile(r'@\w+')
    return pattern.sub('person', text)
#####################################################################

# Abbreviations replacement
def replace_abbreviations(text):
    abbreviation_mapping = {
    'OMG': 'oh my god',
    'DM': 'direct message',
    'BTW': 'by the way',
    'BRB': 'be right back',
    'RT': 'retweet',
    'FTW': 'for the win',
    'QOTD': 'quote of the day',
    'IDK': 'I do not know',
    'ICYMI': 'in case you missed it',
    'IRL': 'in real life',
    'IMHO': 'in my humble opinion',
    'IMO': 'I do not know',
    'LOL': 'laugh out loud',
    'LMAO': 'laughing my ass off',
    'LMFAO': 'laughing my fucking ass off',
    'NTS': 'note to self',
    'F2F': 'face to face',
    'B4': 'before',
    'CC': 'carbon copy',
    'SMH': 'shaking my head',
    'STFU': 'shut the fuck up',
    'BFN': 'by for now',
    'AFAIK': 'as far as I know',
    'TY': 'thank you',
    'YW': 'you are welcome',
    'THX': 'thanks'
}

    pattern = re.compile(r'\b(' + '|'.join(re.escape(abbreviation) for abbreviation in abbreviation_mapping.keys()) + r')\b', re.IGNORECASE)

    #replace abbreviations with its full form
    def replace(match):
        #preserving the original case
        return abbreviation_mapping[match.group().upper()]

    #replace all matches in the text
    return pattern.sub(replace, text)




###############################################################

# Put preprocessed text in JSON
def turnToJSON(dataset):
    data_dict = dataset.to_dict(orient='records')

    with open('Final_Model/Twitter/Preprocessed.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

#################################################################

#lemin