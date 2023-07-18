class Document:
    def __init__(self, id, text, taxonomy):
        self.id = id
        self.text = text
        self.taxonomy = taxonomy

    @property
    def view_info(self):
        return {
            "id": str(self.id),
            "text": self.text,
            "taxonomy": self.taxonomy
        }
