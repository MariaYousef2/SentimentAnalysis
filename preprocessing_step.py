

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from farasa.pos import FarasaPOSTagger
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize Farasa tools
farasa_segmenter = FarasaSegmenter(interactive=True)
farasa_stemmer = FarasaStemmer(interactive=True)

# Function to remove diacritics
def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

# Function to handle hashtags
def handle_hashtags(text):
    return re.sub(r'#\w+', '', text)

# Function to handle emojis
def handle_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Function to remove Twitter metadata
def remove_twitter_metadata(text):
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'RT[\s]+', '', text)  # Remove retweets
    return text

# Function to remove special characters
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Function to remove newlines
def remove_newlines(text):
    return text.replace('\n', ' ')

# Function to tokenize text
def tokenize(text):
    return nltk.word_tokenize(text)

# Function to remove stopwords
def remove_stopwords(tokens):
    arabic_stopwords = set(stopwords.words('arabic'))
    return [word for word in tokens if word not in arabic_stopwords]

# Function to stem text
def stem_text(tokens):
    return [farasa_stemmer.stem(word) for word in tokens]

# Read the CSV file with a different encoding
file_path = 'hh.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Preprocess the tweets
def preprocess_tweet(tweet):
    tweet = remove_diacritics(tweet)
    tweet = handle_hashtags(tweet)
    tweet = handle_emojis(tweet)
    tweet = remove_twitter_metadata(tweet)
    tweet = remove_special_characters(tweet)
    tweet = remove_newlines(tweet)
    tokens = tokenize(tweet)
    tokens = remove_stopwords(tokens)
    tokens = stem_text(tokens)
    return ' '.join(tokens)

df['processed_tweet'] = df.iloc[:, 0].apply(preprocess_tweet)

# Display the first few lines of the processed DataFrame
df.head()

