# Import built-in libraries
import logging
from typing import List

# Import third-party libraries
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SentenceTokenizer():
    def tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text)


class WordTokenizer():
    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)
        
