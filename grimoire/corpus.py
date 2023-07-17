# Import native libraries
import logging
import pandas as pd
import pickle
import random
import threading
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
        
        logging.info(f"Created a new Corpus instance with ID: {self.id} at {self.created_date}")

    def view_info(self):
        return {
            "id": self.id,
            "created_date": self.created_date,
            "num_documents": len(self.documents)
        }
     
    def add_documents(self, document_ids, username, password, batch_size):
        BATCH_SIZE = batch_size
        num_batches = len(document_ids) // BATCH_SIZE

        document_ids = []
        document_metadata = []
        document_contents = []

        def get_access_token(domain, username, password):
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
        
        def get_document_metadata(unique_id, token):
            url = f"{DOCLINK_METADATA_URL}".format(unique_id)
            payload = {}
            headers = {
                "Accept": "application/json",
                "Authorization": "Bearer" + token
            }
            response = requests.get(url=url, headers=headers, data=payload)
            return response.json()
        
        def get_document_text(unique_id, token):
            url = f"{DOCLINK_TEXT_URL}".format(unique_id)
            payload = {}
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer" + token
            }
            response = requests.get(url=url, headers=headers, data=payload)
            return response.text

        def process_batch(self, batch_ids, batch_num, num_batches, username, password):
            token = get_access_token(username, password)
            logging.info(f"Started processing batch {batch_num}/{num_batches}")

            try:
                batch_metadata = [get_document_metadata(document_id, token) for document_id in batch_ids]
                batch_contents = [get_document_text(document_id, token) for document_id in batch_ids]

                document_ids.extend(batch_ids)
                document_metadata.extend(batch_metadata)
                document_contents.extend(batch_contents)
            except Exception as e:
                logging.error(f"Exception occurred while downloading batch {batch_num}: {e}")

        threads = []
        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            batch_ids = document_ids[start:end]

            t = threading.Thread(target=process_batch, args=(batch_ids, i + 1))
            threads.append(t)
            t.start()

        remaining_ids = document_ids[num_batches * BATCH_SIZE:]
        if remaining_ids:
            logging.info("Started processing remaining documents")
            token = get_access_token(username, password)
            try:
                remaining_metadata = [get_document_metadata(remaining_id, token) for remaining_id in remaining_ids]
                remaining_contents = [get_document_text(remaining_id, token) for remaining_id in remaining_ids]

                document_ids.extend(remaining_ids)
                document_metadata.extend(remaining_metadata)
                document_contents.extend(remaining_contents)
            except Exception as e:
                logging.error(f"Exception occurred while downloading remaining documents: {e}")
            
            logging.info("Finished processing all documents")

        self.df = pd.DataFrame({
            "Document_ID": document_ids,
            "Document_Metadata": document_metadata,
            "Document_Contents": document_contents
        })

        self.df

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
