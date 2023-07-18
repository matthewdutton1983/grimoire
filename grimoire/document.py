class Document:
    def __init__(self, id, text, attributes):
        self.id = id
        self.text = text
        self.attributes = attributes

    @property
    def view_info(self):
        return {
            "id": str(self.id),
            "text": self.text,
            "attributes": self.attributes
        }
