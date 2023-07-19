# Import native libraries
from typing import Any, Dict


class Document:
    """
    This class represents a Document in the Corpus. It contains the id, text, and attributes of the document.
    """

    def __init__(self, id: str, text: str, attributes: Dict[str, Any]) -> None:
        """
        Initializes a new Document instance.

        Args:
            id (str): The id of the document.
            text (str): The text content of the document.
            attributes (Dict[str, Any]): The attributes of the document.
        """
        self.id = id
        self.text = text
        self.attributes = attributes

    @property
    def view_info(self) -> Dict[str, Any]:
        """
        Returns the metadata of the document.

        Returns:
            Dict[str, Any]: A dictionary with the id, text, and attributes of the document.
        """
        return {
            "id": str(self.id),
            "text": self.text,
            "attributes": self.attributes
        }
