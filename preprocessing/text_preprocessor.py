import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean social media text.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove special characters and emojis
    # Keep # for hashtags
    text = re.sub(r"[^A-Za-z0-9\s#]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


def tokenize_text(text):
    """
    Convert cleaned text into individual words.
    """
    return word_tokenize(text)


def remove_stopwords(tokens):
    """
    Remove common English stopwords
    and standalone hashtag symbols.
    """

    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and word != "#"
    ]

    return filtered_tokens


def stem_tokens(tokens):
    """
    Apply stemming to tokens.
    """
    return [
        stemmer.stem(word)
        for word in tokens
    ]


def lemmatize_tokens(tokens):
    """
    Apply lemmatization to tokens.
    """
    return [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]


def preprocess_text(text):
    """
    Complete preprocessing pipeline:

    Raw Text
        ↓
    Cleaning
        ↓
    Tokenization
        ↓
    Stopword Removal
        ↓
    Stemming
        ↓
    Lemmatization
        ↓
    Final Tokens
    """

    cleaned_text = clean_text(text)

    tokens = tokenize_text(cleaned_text)

    filtered_tokens = remove_stopwords(tokens)

    stemmed_tokens = stem_tokens(filtered_tokens)

    lemmatized_tokens = lemmatize_tokens(stemmed_tokens)

    return lemmatized_tokens