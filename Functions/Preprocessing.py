#All preprocessing functions

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import emoji
import json


import text_mining_utils as tmu


def remove_Sarcasm_hashtag():
    file_path = 'Datasets/Train_v1.txt'
    column_names = ['toRemove', 'isSarcastic', 'text']

    #read the dataset
    data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

    #define patterns to remove
    patterns = [
        r'#sarcasm\b',
        r'#not\b',
        r'#Not\b',
        r'#sarcastic\b',
        r'#yeahright'

    ]

    #remove patterns from text
    for pattern in patterns:
        data['text'] = data['text'].apply(lambda x: re.sub(pattern, '', x))

    #drop the 'toRemove' column
    data.drop(columns=['toRemove'], inplace=True)

    #convert DataFrame to dictionary
    data_dict = data.to_dict()

    #save the dictionary to a JSON file
    with open('Final_Model/Twitter/cleaned_sarcasm.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

#remove_Sarcasm_hashtag()

#######################################################################################

#undersampling
def undersample(dataset, colName):
    class_counts = dataset[colName].value_counts()

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    minority_class_count = class_counts[minority_class]

    majority_class_sampled = dataset[dataset[colName] == majority_class].sample(n=minority_class_count, random_state=42)

    balanced_data = pd.concat([majority_class_sampled, dataset[dataset[colName] == minority_class]])

    #shuffle data
    balanced_data = balanced_data.sample(frac=1, random_state=1).reset_index(drop=True)
    return balanced_data

#####################################################

#random oversampling
def oversample(text, label, x_name, y_name):
    text = text.values.reshape(-1, 1)
    
    #define and apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(text, label)
    
    X_res = X_res.flatten()
    
    #create and return the new dataframe
    oversampled_dataset = pd.DataFrame({
        x_name: X_res,
        y_name: y_res
    })

    return oversampled_dataset


################################################

#remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

######################################################

#emoji and emoticon replacement

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

    #function to replace match with corresponding word
    def replace_func(match):
        return emoticon_dict_lower.get(match.group().lower(), match.group())

    return pattern.sub(replace_func, text)

def replaceEmoji_emoticon(dataset):
    dataset.text = dataset.text.apply(replace_emoticons)
    dataset.text = dataset.text.apply(replace_emojis)
    return dataset

######################################################################

#remove @usersname
def remove_user_mentions(text):
    pattern = re.compile(r'@\w+')
    return pattern.sub('person', text)
#####################################################################

#abbreviations replacement
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

    #replace abbreviations with full form
    def replace(match):
        return abbreviation_mapping[match.group().upper()]

    #replace all matches
    return pattern.sub(replace, text)




###############################################################

#put preprocessed text in json
def turnToJSON(dataset):
    data_dict = dataset.to_dict(orient='records')

    with open('Final_Model/Twitter/Preprocessed.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

###############################################################

#replace contractions
def contractions_replaced(dataset , col_name):
    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "might've": "might have",
        "must've": "must have",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that'll": "that will",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'd": "who would",
        "who'll": "who will",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why'll": "why will",
        "why's": "why is",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    #use text analysis function aurellia provided
    dataset[col_name] = dataset[col_name].apply(lambda x: tmu.resolve_contractions(x, contractions_dict))
    
    return dataset

######################################################################