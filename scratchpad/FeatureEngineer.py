import logging
import nltk
import spacy
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download("punkt")
nltk.download("stopwords")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')


class FeatureEngineer:
    def __init__(self):
        logging.info('Initializing FeatureEngineer...')
        self.nlp = spacy.load('en_core_web_sm')

    def load_text(self, text):
        logging.info("Loading text...")
        self.text = text

    def tokenize(self, text):
        logging.info('Tokenizing text...')
        tokens = []
        for space_separated_fragment in text.strip().split():
            tokens.extend(nltk.word_tokenize(space_separated_fragment))
            tokens.append(' ')
        return tokens[:-1]

    @property
    def original_text(self, text):
        return text

    @property
    def vocab(self):
        return dict((word, i) for i, word in enumerate(set(self.tokens)))

    @property
    def count_bow(self):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([' '.join(self.tokens)])
        return dict(zip(vectorizer.get_feature_names(), X.toarray()[0]))

    @property
    def tfidf_bow(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([' '.join(self.tokens)])
        return dict(zip(vectorizer.get_feature_names(), X.toarray()[0]))

    @property
    def word_frequency(self):
        words_only = [
            token for token in self.tokens if token not in string.punctuation and not token.isspace()]
        word_counts = Counter(words_only)
        return dict(word_counts)

    @property
    def average_word_length(self):
        words = [token for token in self.tokens if token.isalpha()]
        if len(words) == 0:  # Avoid division by zero if there are no words.
            return 0
        return sum(len(word) for word in words) / len(words)

    def is_special_character(self, token):
        return any(c in string.punctuation for c in token)

    def token_case(self, token):
        if token.islower():
            return 'lowercase'
        elif token.isupper():
            return 'uppercase'
        elif token.istitle():
            return 'capitalized'
        else:
            return 'mixed'

    def is_alpha(self, token):
        return token.isalpha()

    def is_digit(self, token):
        return token.isdigit()

    def is_linefeed(self, token):
        return token == '\n'

    def is_stopword(self, token):
        return token.lower() in nltk.corpus.stopwords.words('english')

    def pos_tags(self):
        doc = self.nlp(' '.join(self.tokens))
        return dict((token.text, token.pos_) for token in doc)

    def char_offsets(self, token):
        start = self.text.find(token)
        end = start + len(token)
        return [start, end]

    def extract_features(self):
        logging.info('Extracting features from text...')
        features = []
        for i, token in enumerate(self.tokens):
            features.append({
                'token': token,
                'is_special_character': self.is_special_character(token),
                'casing': self.token_case(token),
                'length': len(token),
                'pos_tag': self.pos_tag_dict.get(token, None),
                'is_alpha': self.is_alpha(token),
                'is_digit': self.is_digit(token),
                'is_linefeed': self.is_linefeed(token),
                'is_stopword': self.is_stopword(token),
                'offsets': self.char_offsets(token),
                'lowercase': token.lower()
            })
        return features
