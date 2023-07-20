# Import native libraries
import os
from datetime import datetime

# Import third-party libraries
import spacy

# Import project code
from grimoire.nlp.features import Features

nlp = spacy.load("/models/en_core_web_lg-3.4.0")


class Document:
    def __init__(self, id, text, attributes):
        self.id = id
        self.date_added = str(datetime.now())
        self.added_by = os.getlogin()
        self.text = text
        self.attributes = attributes    
        self.features = Features()
        self.vocabulary = []

    def extract_features(self):
        self.features = nlp(self.text)

    def build_vocabulary(self, tokenizer):
        tokens = tokenizer.tokenize(self.text)
        words = [token.lower() for token in tokens if token.isalpha()]
        self.vocabulary = sorted(list(set(words)))
