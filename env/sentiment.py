# importing the libraries
import streamlit as st
import joblib
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import contractions
import pandas as pd
import xgboost

# preprocessing functions

# remove punctuations
exclude = string.punctuation
def remove_punctuations(tweet):
    return tweet.translate(str.maketrans('','',exclude))

# tokenize the tweets
def tokenize_text(tweet):
    return word_tokenize(tweet)

# removing stop words
stop_words = set(stopwords.words('english'))
def remove_stopwords(tweet):
    # Use list comprehension for efficient list creation
    new_tweet = [word for word in tweet if word not in stop_words]
    return " ".join(new_tweet)

# Lemmatizer.
word_lem = WordNetLemmatizer()
def lem_words(tweet):
    return [word_lem.lemmatize(word) for word in tweet]

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def process(tweets):
    processed_tweet = []

    # Defining regex patterns.

    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in tweets:

        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        processed_tweet.append(tweet)

    return processed_tweet

# removing the html tags
def remove_html(review):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', review)

# removing URL and @ sign

def preprocess_text_removingq_URLand_atsign(text):
    # Remove URLs
    clean_text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'@[^\s]+', 'user', clean_text)
    # Other preprocessing steps like removing punctuation, converting to lowercase, etc.
    # ...
    return text

# expan
def expand(text):
    # Expand contractions
    expanded_text = contractions.fix(text)
    return expanded_text

# loading the models

binary_model = joblib.load('logistic_regression_model.pkl')
multi_model = joblib.load('xgb.pkl')

# we'll define 2 functions:
# 1. prediction generator.
# 2. the main function.

def prediction_generator_binary(text):

    """function that takes the preprocessed text and passes to model
    to see the sentiment prediction"""

    binary_predict = binary_model.predict(text)

    if binary_predict == 1:
        output_binary = 'Negative sentiment'
    else:
        output_binary = 'Positive sentiment'

    return output_binary

def prediction_generator_multi(text):

    multi_predict = multi_model.predict(text)
    if multi_predict == 1:
        output_multi = 'Negative sentiment'
    elif multi_predict == 3:
        output_multi = 'Positive sentiment'
    elif multi_predict == 2:
        output_multi = 'no emotion toward brand or product'
    else:
        output_multi = 'I cannot tell'
    return output_multi


# main function

def main():
    """handles the layout of the web app"""

    st.title('Testing binary and multi-class sentiment models')

    # tweet input
    tweet = st.text_input('Enter tweet here')

    # tweet preprocessing

    # lowecase
    tweet = tweet.lower()
    # removing html tags
    tweet = remove_html(tweet)
    # removing URL and @ sign
    tweet = preprocess_text_removingq_URLand_atsign(tweet)
    # expanding word
    tweet = expand(tweet)
    # removing punctuations
    tweet = remove_punctuations(tweet)
    # tokenizing the data
    tweet= tokenize_text(tweet)
    # removing stopwords
    tweet = remove_stopwords(tweet)
    # re-tokenizing
    tweet = tokenize_text(tweet)
    # lemmatization
    tweet = lem_words(tweet)
    tweet = ' '.join(tweet)
    tweet = [tweet]

    # vectorizing the tweet for binary model
    data = pd.read_csv('binary.csv')
    bow2 = CountVectorizer()
    bow2.fit_transform(data['lemmatized_review'])
    tweet_x = bow2.transform(tweet)

    # vectorizing the tweet for multi model
    data2 = pd.read_csv('multi.csv')
    bow = CountVectorizer()
    bow.fit_transform(data2['lemmatized_review'])
    tweet_multi = bow.transform(tweet)


    if st.button('Get result'):
        test_result_binary = prediction_generator_binary(tweet_x)
        test_result_multi = prediction_generator_multi(tweet_multi)

        st.success('Binary model says its a {0}'.format(test_result_binary))
        st.success('Multi class model says its a {0}'.format(test_result_multi))

if __name__ == '__main__':
    main()