# Import native libraries
import logging
from typing import List, Tuple

# Import third-party libraries
import requests
from requests.exceptions import RequestException
from retry import retry

# Import project code
from grimoire.config import CLIENT_ID, IDA_URL, RESOURCE, DOCLINK_METADATA_URL, DOCLINK_TEXT_URL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DoclinkConnector:
    """
    This class provides methods for connecting to Doclink and retrieving document data.
    """

    def __init__(self):
        """
        Initializes a new DoclinkConnector instance.
        """
        logging.info("Created new DoclinkConnector instance")

    @retry(RequestException, tries=3, delay=2, backoff=2)
    def get_access_token(self, domain: str, username: str, password: str) -> str:
        """
        Retrieves an access token using the provided domain, username, and password.

        Args:
            domain (str): The domain name.
            username (str): The username.
            password (str): The password.

        Returns:
            str: The access token.
        """
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
    def get_document_metadata(self, unique_id: str, token: str) -> dict:
        """
        Retrieves the metadata of a document given its unique ID and a valid token.

        Args:
            unique_id (str): The unique ID of the document.
            token (str): A valid token.

        Returns:
            dict: The metadata of the document.
        """
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
    def get_document_text(self, unique_id: str, token: str) -> str:
        """
        Retrieves the text content of a document given its unique ID and a valid token.

        Args:
            unique_id (str): The unique ID of the document.
            token (str): A valid token.

        Returns:
            str: The text content of the document.
        """
        url = f"{DOCLINK_TEXT_URL}".format(unique_id)
        payload = {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer" + token
        }
        response = requests.get(url=url, headers=headers, data=payload)
        response.raise_for_status()

        return response.text

    def process_batch(self, batch_ids: List[str], batch_num: int, num_batches: int, domain: str, username: str, password: str) -> Tuple[List[str], List[dict], List[str]]:
        """
        Processes a batch of document IDs to retrieve their metadata and contents.

        Args:
            batch_ids (List[str]): The list of document IDs in the batch.
            batch_num (int): The current batch number.
            num_batches (int): The total number of batches.
            domain (str): The domain name.
            username (str): The username.
            password (str): The password.

        Returns:
            Tuple[List[str], List[dict], List[str]]: A tuple containing the list of IDs, their corresponding metadata, and contents.
        """
        token = self.get_access_token(domain, username, password)
        logging.info(f"Started processing batch {batch_num}/{num_batches}")

        batch_metadata = []
        batch_contents = []

        try:
            batch_metadata = [self.get_document_metadata(
                document_id, token) for document_id in batch_ids]
            batch_contents = [self.get_document_text(
                document_id, token) for document_id in batch_ids]
        except Exception as e:
            logging.error(
                f"Exception occurred while downloading batch {batch_num}: {e}")

        return batch_ids, batch_metadata, batch_contents
