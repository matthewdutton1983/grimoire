# Import native libraries
import concurrent.futures
import logging
import pickle
import random
import uuid
from datetime import datetime

# Import third-party libraries
import pandas as pd

# Import project code
from grimoire.connectors import DoclinkConnector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Corpus:
    def __init__(self, connector: DoclinkConnector):
        self.id = uuid.uuid4()
        self.created_date = datetime.now()
        self.documents = []
        self.df = pd.DataFrame()
        self.connector = connector
        
        logging.info(f"Created new Corpus instance with ID: {self.id} at {self.created_date}")

    @property
    def view_info(self):
        return {
            "id": str(self.id),
            "created_date": self.created_date.isoformat(),
            "num_documents": len(self.df)
        }

    def add_documents(self, document_ids, domain, username, password, batch_size):
        BATCH_SIZE = batch_size
        num_batches = len(document_ids) // BATCH_SIZE

        all_document_ids = []
        all_document_metadata = []
        all_document_contents = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(num_batches):
                start = i * BATCH_SIZE
                end = (i + 1) * BATCH_SIZE
                batch_ids = document_ids[start:end]
                futures.append(executor.submit(self.connector.process_batch, batch_ids, i + 1, num_batches, domain, username, password))

            for future in concurrent.futures.as_completed(futures):
                batch_ids, batch_metadata, batch_contents = future.result()
                all_document_ids.extend(batch_ids)
                all_document_metadata.extend(batch_metadata)
                all_document_contents.extend(batch_contents)

        remaining_ids = document_ids[num_batches * BATCH_SIZE:]
        if remaining_ids:
            logging.info("Started processing remaining documents")
            token = self._get_access_token(domain, username, password)
            try:
                remaining_metadata = [self.connector._get_document_metadata(remaining_id, token) for remaining_id in remaining_ids]
                remaining_content = [self.connector._get_document_text(remaining_id, token) for remaining_id in remaining_ids]

                all_document_ids.extend(remaining_ids)
                all_document_metadata.extend(remaining_metadata)
                all_document_contents.extend(remaining_content)
            except Exception as e:
                logging.error(f"Error occurred while downloading remaining documents")

        logging.info("All documents have been downloaded and added to the corpus")

        self.df = pd.DataFrame({
            "DOCUMENT_ID": all_document_ids,
            "DOCUMENT_METADATA": all_document_metadata,
            "DOCUMENT_CONTENT": all_document_contents
        })

        return self.df
    
    def remove_documents(self, document_ids):
        self.df = self.df[~self.df["DOCUMENT_ID"].isin(document_ids)]
        logging.info(f"Successfully removed documents from the corpus: {document_ids}")
        return self.df

    def random_sample(self, n):
        return random.sample(self.documents, n)

    def save_corpus(self, filename):
        with open(filename, "wb") as f:
            return pickle.dump(self, f)

    @staticmethod
    def load_corpus(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    
