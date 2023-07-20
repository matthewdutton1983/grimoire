# Import native libraries
from datetime import datetime


class Document:
    def __init__(self, id, text, attributes):
        self.id = id
        self.date_added = str(datetime.now())
        self.text = text
        self.attributes = attributes
        self.vocabulary = set()
        
    def build_vocabulary(self, tokenizer):
        tokens = tokenizer.tokenize(self.text)
        words = [token.lower() for token in tokens if token.isalpha()]
        self.vocabulary.update(words)
