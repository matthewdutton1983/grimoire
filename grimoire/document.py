# Import native libraries
from datetime import datetime

# Import project code
from grimoire.dictionary import Dictionary


class Document:
    def __init__(self, id, text, attributes):
        self.id = id
        self.date_added = str(datetime.now())
        self.text = text
        self.attributes = attributes
        
    @property
    def view_info(self):
        return {
            "id": str(self.id),
            "date_added": self.date_added,
            "text": self.text,
            "attributes": self.attributes
        }
          
