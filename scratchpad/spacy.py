import pandas as pd
import spacy
from configparser import ConfigParser
from grimoire.connectors import DoclinkConnector
from grimoire.nlp.tokenizers import SentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer

connector = DoclinkConnector()
token = connector.get_access_token(domain, username, password)

document = connector.get_document_text(unique_ids[0], token)
tokenizer = SentenceTokenizer()

nlp = spacy.load("/models/en_core_web_lg-3.4.0/")

spacy_doc = nlp(document)

tokens = [token for token in spacy_doc]
lemma = [token.lemma_ for token in tokens]
syntax = [token.pos_ for token in tokens]
tags = [token.tag_ for token in tokens]
dep = [token.dep_ for token in tokens]
shape = [token.shape_ for token in tokens]
alpha = [token.is_alpha for token in tokens]
stop_word = [token.is_stop for token in tokens]
lower = [str(token).islower() for token in tokens]
upper = [str(token).isupper() for token in tokens]
title = [str(token).istitle() for token in tokens]
numeric = [str(token).isnumeric() for token in tokens]

columns=["TEXT", "LEMMA", "POS", "TAG", "DEP", "SHAPE", "ALPHA", "STOP", "LOWER", "UPPER", "TITLE", "NUMERIC"]
data = zip(tokens, lemma, syntax, tags, dep, shape, alpha, stop_word, lower, upper, title, numeric)
df = pd.DataFrame(data, columns=columns)

chunks = list(spacy_doc.noun_chunks)
chunks

entities = []
for ent in spacy_doc.ents:
    entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

df = pd.DataFrame(entities, columns=["Entity", "Start", "End", "Type"])
df

sentences = [sent.text.strip() for sent in spacy_doc.sents]
len(sentences)

bow_vectorizer = CountVectorizer()
X = bow_vectorizer.fit_transform(sentences)
bow_df = pd.DataFrame(X.toarray(), columns=bow_vectorizer.get_feature_names_out())
bow_df.head()
