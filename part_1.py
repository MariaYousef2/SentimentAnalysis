import re
import string
import emoji
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_selection import SelectKBest, chi2
from pyswarm import pso

with open('arabic_tweets.txt', 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Remove Diacritic
def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                                ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', text)
    return text
# Handle Hashtags
def handle_hashtags(text):
    return re.sub(r'#\w+', '', text)
# Handle Emojis
def handle_emojis(text):
    return emoji.get_emoji_regexp().sub(u'', text)
# Remove Twitter Metadata
def remove_metadata(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    return text
# Remove Special Characters
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
# Remove Newlines
def remove_newlines(text):
    return text.replace('\n', ' ')
# Tokenization
def tokenize(text):
    return text.split()
# Remove Stopwords
stop_words = set(stopwords.words('arabic'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]
# Stemming
stemmer = ISRIStemmer()
def stem(tokens):
    return [stemmer.stem(word) for word in tokens]
# Apply preprocessing steps
def preprocess(text):
    text = remove_diacritics(text)
    text = handle_hashtags(text)
    text = handle_emojis(text)
    text = remove_metadata(text)
    text = remove_special_characters(text)
    text = remove_newlines(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return ' '.join(tokens)
preprocessed_tweets = [preprocess(tweet) for tweet in tweets]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_tweets)
y = [0] * len(preprocessed_tweets)  # Dummy target variable

def objective_function(x):
    selected_features = [i for i in range(len(x)) if x[i] > 0.5]
    if len(selected_features) == 0:
        return float('inf')
    X_selected = X[:, selected_features]
    chi2_score, _ = chi2(X_selected, y)
    return -chi2_score.sum()
num_features = X.shape[1]
lb = [0] * num_features
ub = [1] * num_features
xopt, fopt = pso(objective_function, lb, ub, swarmsize=30, maxiter=100)
selected_features = [i for i in range(len(xopt)) if xopt[i] > 0.5]
print(f"Number of features selected by PSO: {len(selected_features)}")

