# Import native libraries
import logging

# Import third-party libraries
import requests
from requests.exceptions import RequestException
from retry import retry

# Import project code
from grimoire.config import CLIENT_ID, IDA_URL, RESOURCE, DOCLINK_METADATA_URL, DOCLINK_TEXT_URL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DoclinkConnector:
    def __init__(self):
        logging.info("Created new DoclinkConnector instance")

    @retry(RequestException, tries=3, delay=2, backoff=2)
    def get_access_token(self, domain, username, password):
            url = IDA_URL
            payload = {
                 "client_id": CLIENT_ID,
                 "grant_type": "password",
                 "username": domain + "\\" + username,
                 "password": password,
                 "resource": RESOURCE
            }
            response = requests.post(url, payload)
            response.raise_for_status()
            
            return response.json()["access_token"]
    
    @retry(RequestException, tries=3, delay=2, backoff=2)
    def get_document_metadata(self, unique_id, token):
        url = f"{DOCLINK_METADATA_URL}".format(unique_id)
        payload = {}
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer" + token
        }
        response = requests.get(url=url, headers=headers, data=payload)
        response.raise_for_status()
        
        return response.json()
    
    @retry(RequestException, tries=3, delay=2, backoff=2)
    def get_document_text(self, unique_id, token):
        url = f"{DOCLINK_TEXT_URL}".format(unique_id)
        payload = {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer" + token
        }
        response = requests.get(url=url, headers=headers, data=payload)
        response.raise_for_status()
        
        return response.text
    
    def process_batch(self, batch_ids, batch_num, num_batches, domain, username, password):
        token = self.get_access_token(domain, username, password)
        logging.info(f"Started processing batch {batch_num}/{num_batches}")
        
        batch_metadata = []
        batch_contents = []

        try:
            batch_metadata = [self.get_document_metadata(document_id, token) for document_id in batch_ids]
            batch_contents = [self.get_document_text(document_id, token) for document_id in batch_ids]
        except Exception as e:
            logging.error(f"Exception occurred while downloading batch {batch_num}: {e}")

        return batch_ids, batch_metadata, batch_contents
