# Import third-party libraries
import spacy

model = spacy.load("/models/en_core_web_lg-3.4.0/")


class Features:
    def __init__(self):
        self.tokens = []
        self.lemma = []
        self.syntax = []
        self.tags = []
        self.dep = []
        self.shape = []
        self.alpha = []
        self.stopword = []
        self.lowercase = []
        self.uppercase = []
        self.titlecase = []
        self.numeric = [] 

    def extract_features(self, document):
        self.nlp = model(document.text)
        self.tokens = [token for token in self.nlp]
        self.lemma = [token.lemma_ for token in self.tokens]
        self.syntax = [token.pos_ for token in self.tokens]
        self.tags = [token.tag_ for token in self.tokens]
        self.dep = [token.dep_ for token in self.tokens]
        self.shape = [token.shape_ for token in self.tokens]
        self.alpha = [token.is_alpha for token in self.tokens]
        self.stopword = [token.is_stop for token in self.tokens]
        self.lowercase = [str(token).islower() for token in self.tokens]
        self.uppercase = [str(token).isupper() for token in self.tokens]
        self.titlecase = [str(token).istitle() for token in self.tokens]
        self.numeric = [str(token).isnumeric() for token in self.tokens]
