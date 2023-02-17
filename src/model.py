import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import joblib

def procces_tweet(tweet: str) -> str:
    # Remove all mentions, hashtags, links, and special characters
    tweet = re.sub(r'http\S+|www\S+|https\S+|@\S+|#\S+', '', tweet, flags=re.MULTILINE)
    tweet = tweet.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    tweet = tweet.replace('ة', 'ه').replace('ى', 'ي').replace('ؤ', 'و')

    # Remove all emojis and emoticons and replace them with unicode
    tweet = emoji.demojize(tweet)
    
    # Tokinize the tweet
    tokens = word_tokenize(tweet)
    
    # Remove all stop words, stop words are words that do not add any meaning to the sentence
    stop_words = set(stopwords.words('arabic'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    # Stemming, stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form
    stemmer = SnowballStemmer('arabic')
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    
    return " ".join(stemmed_tokens)