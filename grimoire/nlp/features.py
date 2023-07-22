# Import native libraries
import logging

# Import third-party libraries
import pandas as pd
import spacy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Features:
    logger.info("Loading NLP model ... ")
    nlp = spacy.load("/models/en_core_web_lg-3.4.0/")

    def __init__(self):
        self.noun_chunks = []
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
        self.entities = [] 


    @classmethod
    def extract_features(cls, text):
        features = cls()
        doc = cls.nlp(text)
        
        logger.info("Extracting features ...")
        
        features.noun_chunks = list(doc.noun_chunks)
        features.tokens = [token for token in doc]
        features.lemma = [token.lemma_ for token in features.tokens]
        features.syntax = [token.pos_ for token in features.tokens]
        features.tags = [token.tag_ for token in features.tokens]
        features.dep = [token.dep_ for token in features.tokens]
        features.shape = [token.shape_ for token in features.tokens]
        features.alpha = [token.is_alpha for token in features.tokens]
        features.stopword = [token.is_stop for token in features.tokens]
        features.lowercase = [str(token).islower() for token in features.tokens]
        features.uppercase = [str(token).isupper() for token in features.tokens]
        features.titlecase = [str(token).istitle() for token in features.tokens]
        features.numeric = [str(token).isnumeric() for token in features.tokens]
        
        for ent in doc.ents:
            features.entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
        
        return features
    
    @classmethod
    def summarize_features(cls):
        features = cls()
        columns = ["TOKEN", "LEMMA", "POS", "TAG", "DEP", "SHAPE", "ALPHA", "STOP", "LOWER", "UPPER", "TITLE", "NUMERIC"]
        data = zip(features.tokens, features.lemma, features.syntax, features.tags, features.dep,
            features.shape, features.alpha, features.stopword, features.lowercase,
            features.uppercase, features.titlecase, features.numeric
        )
        df = pd.DataFrame(data, columns=columns)
        return df
