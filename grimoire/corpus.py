# Import native libraries
import concurrent.futures
import logging
import pickle
import random
import uuid
from datetime import datetime

# Import project code
from grimoire.connectors import DoclinkConnector
from grimoire.document import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Corpus:
    def __init__(self, connector = DoclinkConnector):
        self.id = uuid.uuid4()
        self.created_date = datetime.now()
        self.documents = []
        self.connector = connector

        self.__id_to_index = {}
        
        logger.info("Created new Corpus instance")

    @property
    def view_info(self):
        return {
            "id": str(self.id),
            "created_date": self.created_date.isoformat(),
            "num_documents": len(self.documents)
        }

    def add_documents(self, document_ids, domain, username, password):
        BATCH_SIZE = max(1, len(document_ids) // 100)
        num_batches = len(document_ids) // BATCH_SIZE

        all_documents = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
        
            for i in range(num_batches):
                start = i * BATCH_SIZE
                end = (i + 1) * BATCH_SIZE
                batch_ids = document_ids[start:end]
                futures.append(executor.submit(self.connector.process_batch, batch_ids, i + 1, num_batches, domain, username, password))

            for future in concurrent.futures.as_completed(futures):
                batch_ids, batch_metadata, batch_contents = future.result()

                for i in range(len(batch_ids)):
                    all_documents.append(Document(batch_ids[i], batch_contents[i], batch_metadata[i]))

        remaining_ids = document_ids[num_batches * BATCH_SIZE:]

        if remaining_ids:
            logger.info("Started processing remaining documents")
            token = self.connector.get_access_token(domain, username, password)
            
            try:
                remaining_metadata = [self.connector.get_document_metadata(remaining_id, token) for remaining_id in remaining_ids]
                remaining_content = [self.connector.get_document_text(remaining_id, token) for remaining_id in remaining_ids]

                for i in range(len(remaining_ids)):
                    all_documents.append(Document(remaining_ids[i], remaining_content[i], remaining_metadata[i]))
            except Exception as e:
                logger.error(f"Error occurred while downloading remaining documents")

        logger.info("All documents have been downloaded and added to the corpus")

        self.documents.extend(all_documents)

        for i, doc in enumerate(all_documents):
            self.__id_to_index[doc.id] = len(self.documents) - len(all_documents) + i
    
    def get_document_by_id(self, document_id):
        if document_id in self.__id_to_index:
            return self.documents[self.__id_to_index[document_id]]
        else:
            logging.error(f"No document found with ID: {document_id}")
            return None
        
    def remove_documents(self, document_ids):
        self.documents = [doc for doc in self.documents if doc.id not in document_ids]
        self.__id_to_index = {id: i for i, id in enumerate(doc.id for doc in self.documents)}
        logger.info(f"Successfully removed documents from corpus: {document_ids}")

    def search_corpus(self, query):
        return [doc for doc in self.documents if query.lower() in doc.content.lower()]
    
    def random_sample(self, n):
        return random.sample(self.documents, n)

    def save_corpus(self):
        with open(self.id, "wb") as f:
            return pickle.dump(self, f)

    @staticmethod
    def load_corpus(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
        
