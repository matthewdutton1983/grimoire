# Import native libraries
import concurrent.futures
import logging
import pickle
import random
import uuid
from datetime import datetime
from typing import List, Optional, Type

# Import project code
from grimoire.connectors.doclink import DoclinkConnector
from grimoire.document import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Corpus:
    """
    This class represents a Corpus of Documents. It contains methods for adding and removing 
    documents, searching the corpus, and saving/loading the corpus to/from disk.
    """

    def __init__(self, connector: Type[DoclinkConnector] = DoclinkConnector) -> None:
        """
        Initializes a new Corpus instance.

        Args:
            connector (Type[DoclinkConnector], optional): Connector class for fetching documents. Defaults to DoclinkConnector.
        """
        self.id = uuid.uuid4()
        self.created_date = datetime.now()
        self.documents: List[Document] = []
        self.connector = connector

        self.__id_to_index = {}

        logging.info(f"Created new Corpus instance: {self.id}")

    @property
    def view_info(self) -> dict:
        """
        Returns the metadata of the corpus.

        Returns:
            dict: A dictionary with the id, created_date, and number of documents in the corpus.
        """
        return {
            "id": str(self.id),
            "created_date": self.created_date.isoformat(),
            "num_documents": len(self.documents)
        }

    def add_documents(self, document_ids: List[str], domain: str, username: str, password: str) -> None:
        """
        Fetches and adds documents to the corpus.

        Args:
            document_ids (List[str]): A list of document ids to fetch.
            domain (str): The domain for accessing the documents.
            username (str): The username for accessing the documents.
            password (str): The password for accessing the documents.
        """
        BATCH_SIZE = max(1, len(document_ids) // 100)
        num_batches = len(document_ids) // BATCH_SIZE

        all_documents = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for i in range(num_batches):
                start = i * BATCH_SIZE
                end = (i + 1) * BATCH_SIZE
                batch_ids = document_ids[start:end]
                futures.append(executor.submit(self.connector.process_batch,
                               batch_ids, i + 1, num_batches, domain, username, password))

            for future in concurrent.futures.as_completed(futures):
                batch_ids, batch_metadata, batch_contents = future.result()

                for i in range(len(batch_ids)):
                    all_documents.append(
                        Document(batch_ids[i], batch_contents[i], batch_metadata[i]))

        remaining_ids = document_ids[num_batches * BATCH_SIZE:]

        if remaining_ids:
            logging.info("Started processing remaining documents")
            token = self.connector.get_access_token(domain, username, password)

            try:
                remaining_metadata = [self.connector.get_document_metadata(
                    remaining_id, token) for remaining_id in remaining_ids]
                remaining_content = [self.connector.get_document_text(
                    remaining_id, token) for remaining_id in remaining_ids]

                for i in range(len(remaining_ids)):
                    all_documents.append(
                        Document(remaining_ids[i], remaining_content[i], remaining_metadata[i]))
            except Exception as e:
                logging.error(
                    f"Error occurred while downloading remaining documents")

        logging.info(
            "All documents have been downloaded and added to the corpus")

        self.documents.extend(all_documents)

        for i, doc in enumerate(all_documents):
            self.__id_to_index[doc.id] = len(
                self.documents) - len(all_documents) + i

    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieves a document from the corpus by its id.

        Args:
            document_id (str): The id of the document to retrieve.

        Returns:
            Optional[Document]: The document if it exists, else None.
        """
        if document_id in self.__id_to_index:
            return self.documents[self.__id_to_index[document_id]]
        else:
            logging.error(f"No document found with ID: {document_id}")
            return None

    def remove_documents(self, document_ids: List[str]) -> None:
        """
        Removes documents from the corpus.

        Args:
            document_ids (List[str]): A list of document ids to remove.
        """
        self.documents = [
            doc for doc in self.documents if doc.id not in document_ids]
        self.__id_to_index = {id: i for i, id in enumerate(
            doc.id for doc in self.documents)}
        logging.info(
            f"Successfully removed documents from corpus: {document_ids}")

    def search_corpus(self, query: str) -> List[Document]:
        """
        Searches the corpus for a query string.

        Args:
            query (str): The query string to search for.

        Returns:
            List[Document]: A list of documents where the query string is found.
        """
        return [doc for doc in self.documents if query.lower() in doc.content.lower()]

    def random_sample(self, n):
        """
        Returns a random sample of n documents from the corpus.

        Args:
            n (int): The number of documents to sample.

        Returns:
            List[Document]: A list of n randomly sampled documents.
        """
        return random.sample(self.documents, n)

    def save_corpus(self, filename: str) -> None:
        """
        Saves the corpus to a file.

        Args:
            filename (str): The name of the file to save the corpus to.
        """
        with open(filename, "wb") as f:
            return pickle.dump(self, f)

    @staticmethod
    def load_corpus(filename: str) -> 'Corpus':
        """
        Loads a corpus from a file.

        Args:
            filename (str): The name of the file to load the corpus from.

        Returns:
            Corpus: The loaded corpus.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
