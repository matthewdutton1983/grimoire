# Import native libraries
import os
from datetime import datetime

# Import project code
from grimoire.nlp.features import Features


class Document:
    def __init__(self, id, text, attributes):
        self.id = id
        self.date_added = str(datetime.now())
        self.added_by = os.getlogin()
        self.text = text
        self.attributes = attributes
        self.features = Features()    

    def extract_features(self):
        self.features = Features.extract_features(self.text)
