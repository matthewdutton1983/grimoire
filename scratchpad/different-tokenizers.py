def tokenizer(cls):
    tokenizer.tokenizers.append(cls)
    return cls


tokenizer.tokenizers = []

# Import built-in libraries
import logging
from abc import ABC, abstractmethod
from typing import Callable, List

# Import third-party libraries
import spacy
import stanza
from flair.data import Sentence
from gensim.utils import simple_preprocess
from keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import sent_tokenize, word_tokenize
from pattern.text.en import parse as pattern_parse
from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions, word_tokenizer
from spacy.language import Language
from torchtext.data import get_tokenizer as torch_tokenizer
from transformers import AutoTokenizer
from textblob import TextBlob

# Import project code
from .decorators import tokenizer

logger = logging.getLogger(__file__)


class Tokenizer(ABC):
    """
    Abstract class representing a :class:'Tokenizer'
    """

    def __init__(self):
        self.available_tokenizers = [
            cls.__name__ for cls in tokenizer.tokenizers]

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Get the name of the Tokenizer class.

        Returns:
            str: Name of the class.
        """
        return self.__class__.__name__


@tokenizer
class FlairTokenizer(Tokenizer):
    """
    Tokenizer using Flair under the hood.
    """

    def __init__(self) -> None:
        """
        Initializes the FlairTokenizer instance.
        """
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using Flair.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        sentence = Sentence(text)
        return [str(token) for token in sentence]


@tokenizer
class GensimTokenizer(Tokenizer):
    """
    Tokenizer using Gensim under the hood.
    """

    def __init__(self) -> None:
        """
        Initializes the GensimTokenizer instance.
        """
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using Gensim.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return simple_preprocess(text)


@tokenizer
class HuggingFaceTokenizer(Tokenizer):
    """
    Tokenizer using Hugging Face's transformers library under the hood.

    Args:
        model (str or AutoTokenizer): HF tokenizer instance or the name of the tokenizer to load.
    """

    def __init__(self, model) -> None:
        """
        Initializes the HuggingFaceTokenizer instance.
        """
        super().__init__()

        if isinstance(model, str):
            self.model = AutoTokenizer.from_pretrained(model)
        elif isinstance(model, AutoTokenizer):
            self.model = model
        else:
            raise AssertionError("Unexpected type of parameter model")

    def tokenize(self, text: str, encode: bool = False) -> List[str]:
        """
        Tokenize the input text using Hugging Face's transformers.

        Args:
            text (str): Input text.
            encode (bool): Flag to indicate whether to return tokens or encoded token IDs. 
                           If True, returns encoded token IDs, otherwise, returns tokens.

        Returns:
            List[str]: List of tokens or token IDs based on the encode flag.
        """
        try:
            if encode:
                return self.model.encode(text)
            else:
                encoded_input = self.model.encode(text)
                return self.model.convert_ids_to_tokens(encoded_input)
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise


@tokenizer
class KerasTokenizer(Tokenizer):
    """
    Tokenizer using Keras under the hood.
    """

    def __init__(self) -> None:
        """
        Initializes the KerasTokenizer instance.
        """
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using Keras.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return text_to_word_sequence(text)


@tokenizer
class NLTKTokenizer(Tokenizer):
    """
    Tokenizer using NLTK under the hood.

    Args:
        type (str): Type of tokenizer to use. Default is 'word'.
    """

    def __init__(self, type: str = 'word') -> None:
        """
        Initializes the NLTKTokenizer instance.
        """
        super().__init__()
        self.type = type

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using NLTK.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        try:
            if self.type == 'word':
                return word_tokenize(text)
            elif self.type == 'sentence':
                return sent_tokenize(text)
            else:
                raise ValueError(f'Tokenizer type not recognized: {self.type}')
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise


@tokenizer
class PatternTokenizer(Tokenizer):
    """
    Tokenizer class that uses the Pattern library for tokenization.

    This class inherits from the base Tokenizer class and implements
    the tokenize method using the tokenization functionality provided by Pattern.

    Attributes
    ----------
    None

    Methods
    -------
    tokenize(text: str) -> List[str]:
        Tokenizes the input text into a list of words using Pattern.
    """

    def __init__(self) -> None:
        """
        Initializes the PatternTokenizer instance.
        """
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into a list of words using Pattern.

        Parameters
        ----------
        text : str
            The input text to be tokenized.

        Returns
        -------
        list of str
            The tokenized text as a list of words.
        """
        try:
            return pattern_parse(text, tokenize=True).split()
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise


@tokenizer
class PyTorchTokenizer(Tokenizer):
    """
    Tokenizer using PyTorch under the hood.
    """

    def __init__(self, language='basic_english') -> None:
        """
        Initializes the PyTorchTokenizer instance.
        """
        super().__init__()
        self.tokenizer = torch_tokenizer(language)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using PyTorch.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return self.tokenizer(text)


@tokenizer
class SegtokTokenizer(Tokenizer):
    """
    Tokenizer using segtok under the hood.
    """

    def __init__(self) -> None:
        """
        Initializes the SegtokTokenizer instance.
        """
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using segtok.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return SegtokTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[str]:
        """
        Perform the tokenization of the input text using segtok's rules.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        try:
            words: List[str] = []

            sentences = split_single(text)
            for sentence in sentences:
                contractions = split_contractions(word_tokenizer(sentence))
                words.extend(contractions)

            words = list(filter(None, words))

            return words
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise


@tokenizer
class SpacyTokenizer(Tokenizer):
    """
    Tokenizer using spaCy under the hood.

    Args:
      model (str or Language): A spaCy model or the name of the model to load.
    """

    def __init__(self, model) -> None:
        """
        Initializes the SpacyTokenizer instance.
        """
        super().__init__()

        if isinstance(model, Language):
            self.model = model
        elif isinstance(model, str):
            self.model = spacy.load(model)
        else:
            raise AssertionError("Unexpected type of parameter model")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using spaCy.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        try:
            return [token.text for token in self.model(text)]
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise


@tokenizer
class StanzaTokenizer(Tokenizer):
    """
    Tokenizer using Stanza's tokenization.

    Args:
        lang (str): Language of the text to be tokenized. Default is 'en' (English).
    """

    def __init__(self, lang: str = 'en') -> None:
        """
        Initializes the StanzaTokenizer instance.
        """
        super().__init__()
        self.nlp = stanza.Pipeline(lang=lang, processors='tokenize')

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using Stanza.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        try:
            doc = self.nlp(text)
            tokens = [
                token.text for sent in doc.sentences for token in sent.tokens]
            return tokens
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise


@tokenizer
class TextBlobTokenizer(Tokenizer):
    """
    Tokenizer class that uses the TextBlob library for tokenization.

    This class inherits from the base Tokenizer class and implements
    the tokenize method using the word tokenization functionality provided by TextBlob.

    Attributes
    ----------
    None

    Methods
    -------
    tokenize(text: str) -> List[str]:
        Tokenizes the input text into a list of words using TextBlob.
    """

    def __init__(self) -> None:
        """
        Initializes the TextBlobTokenizer instance.
        """
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into a list of words using TextBlob.

        Parameters
        ----------
        text : str
            The input text to be tokenized.

        Returns
        -------
        list of str
            The tokenized text as a list of words.
        """
        try:
            tb = TextBlob(text)
            return tb.words
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise


class TokenizerWrapper(Tokenizer):
    """
    Helper class to wrap tokenizer functions to the class-based tokenizer interface.

    Args:
        tokenizer_func (Callable[[str], List[str]]): Tokenizer function to wrap.
    """

    def __init__(self, tokenizer_func: Callable[[str], List[str]]) -> None:
        super().__init__()
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using the wrapped function.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        try:
            return self.tokenizer_func(text)
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise

    @property
    def name(self) -> str:
        """
        Get the name of the TokenizerWrapper class and wrapped function.

        Returns:
            str: Name of the class and function.
        """
        return self.__class__.__name__ + "_" + self.tokenizer_func.__name__


@tokenizer
class WhitespaceTokenizer(Tokenizer):
    """Tokenizer based on space character only."""

    def __init__(self) -> None:
        """
        Initializes the SpaceTokenizer instance.
        """
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text based on spaces.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return WhitespaceTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[str]:
        """
        Perform the tokenization of the input text by splitting it at each space character.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        try:
            tokens: List[str] = []
            word = ''
            index = -1
            for index, char in enumerate(text):
                if char == ' ':
                    if len(word) > 0:
                        tokens.append(word)

                    word = ''
                else:
                    word += char
            # increment for last token in sentence if not followed by whitespace
            index += 1
            if len(word) > 0:
                tokens.append(word)

            return tokens
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise
