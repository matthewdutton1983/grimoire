# Import native libraries
import logging

# Import third-party libraries
import pandas as pd
import spacy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Token
# LinguisticParser
# Lemmatizer
# DependencyParser
# NamedEntityParser
# Tokenizer
# SentenceSegmenter
# Embedder
# Preprocessor


class Features:
    logger.info("Loading NLP model ... ")
    nlp = spacy.load("/models/en_core_web_lg-3.4.0/")

    def __init__(self):
        # Document level
        self.noun_chunks = []
        self.entities = [] 

        # Token level
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


    @classmethod
    def extract_features(cls, text):
        logger.info("Creating features ...")
        
        features = cls()
        doc = cls.nlp(text)
        
        # Extract features using spaCy
        features.noun_chunks = list(doc.noun_chunks)
        features.tokens = [token for token in doc]
        features.lemma = [token.lemma_ for token in features.tokens]
        features.syntax = [token.pos_ for token in features.tokens]
        features.tags = [token.tag_ for token in features.tokens]
        features.dep = [token.dep_ for token in features.tokens]
        features.shape = [token.shape_ for token in features.tokens]
        features.alpha = [token.is_alpha for token in features.tokens]
        features.stopword = [token.is_stop for token in features.tokens]

        for ent in doc.ents:
            features.entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

        # Use native Python functions to create additional features
        features.lowercase = [str(token).islower() for token in features.tokens]
        features.uppercase = [str(token).isupper() for token in features.tokens]
        features.titlecase = [str(token).istitle() for token in features.tokens]
        features.numeric = [str(token).isnumeric() for token in features.tokens]
        
        return features
    

    def summarize_features(self):
        columns = ["TOKEN", "LEMMA", "POS", "TAG", "DEP", "SHAPE", "ALPHA", "STOP", "LOWER", "UPPER", "TITLE", "NUMERIC"]
        data = zip(
            self.tokens, self.lemma, self.syntax, self.tags, self.dep, self.shape, self.alpha, 
            self.stopword, self.lowercase, self.uppercase, self.titlecase, self.numeric
        )
        df = pd.DataFrame(data, columns=columns)
        return df
