# Import native libraries
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Dictionary:
    def __init__(self, tokenizer):
        self.id = uuid.uuid4()
        self.date_created = datetime.now()
        self.tokenizer = tokenizer
        self.vocabulary = set()

        logger.info("Created new Dictionary instance")

    @property
    def view_info(self):
        return {
            "id": str(self.id),
            "date_created": self.date_created,
            "tokenizer": self.tokenizer,
            "vocabulary": self.vocabulary
        }
