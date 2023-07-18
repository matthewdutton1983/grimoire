# Import native libraries
import concurrent.futures
import logging
import pandas as pd
import pickle
import random
import uuid
from datetime import datetime

# Import third-party libraries
import pandas as pd
import requests

# Import project code
from grimoire.config import CLIENT_ID, IDA_URL, RESOURCE, DOCLINK_METADATA_URL, DOCLINK_TEXT_URL
from grimoire.dictionary import Dictionary
from grimoire.document import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Corpus:
    def __init__(self):
        self.id = uuid.uuid4()
        self.created_date = datetime.now()
        self.documents = []
        self.df = pd.DataFrame()
        
        logging.info(f"Created new Corpus instance with ID: {self.id} at {self.created_date}")

    @property
    def view_info(self):
        return {
            "id": self.id,
            "created_date": self.created_date,
            "num_documents": len(self.df)
        }
    
    def _get_access_token(self, domain, username, password):
            url = IDA_URL
            payload = dict(
                client_id = CLIENT_ID,
                grant_type = "password",
                username = domain + "\\" + username,
                password = password,
                resource = RESOURCE
            )
            response = requests.post(url, payload)
            token = response.json()["access_token"]
            return token
        
    def _get_document_metadata(self, unique_id, token):
        url = f"{DOCLINK_METADATA_URL}".format(unique_id)
        payload = {}
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer" + token
        }
        response = requests.get(url=url, headers=headers, data=payload)
        return response.json()
    
    def _get_document_text(self, unique_id, token):
        url = f"{DOCLINK_TEXT_URL}".format(unique_id)
        payload = {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer" + token
        }
        response = requests.get(url=url, headers=headers, data=payload)
        return response.text
    
    def _process_batch(self, batch_ids, batch_num, num_batches, domain, username, password):
            token = self._get_access_token(domain, username, password)
            logging.info(f"Started processing batch {batch_num}/{num_batches}")
            
            batch_metadata = []
            batch_contents = []

            try:
                batch_metadata = [self._get_document_metadata(document_id, token) for document_id in batch_ids]
                batch_contents = [self._get_document_text(document_id, token) for document_id in batch_ids]
            except Exception as e:
                logging.error(f"Exception occurred while downloading batch {batch_num}: {e}")

            return batch_ids, batch_metadata, batch_contents

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
                futures.append(executor.submit(self._process_batch, batch_ids, i + 1, num_batches, domain, username, password))

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
                remaining_metadata = [self._get_document_metadata(remaining_id, token) for remaining_id in remaining_ids]
                remaining_content = [self._get_document_text(remaining_id, token) for remaining_id in remaining_ids]

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

    def save_corpus(self, filename):
        with open(filename, "wb") as f:
            return pickle.dump(self, f)

    @staticmethod
    def load_corpus(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
       
    def remove_documents(self):
        pass

    def random_sample(self, n):
        return random.sample(self.documents, n)
