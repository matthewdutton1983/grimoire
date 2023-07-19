import logging
import os
import uuid


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DirectoryConnector:
    def __init__(self):
        self.id_to_file = {}
        logger.info("Created new DirectoryConnector instance")

    def create_document_ids(self, dir_path):
        files = [f for f in os.listdir(
            dir_path) if os.path.isfile(os.path.join(dir_path, f))]

        for file in files:
            id = uuid.uuid4()
            self.id_to_file[str(id)] = os.path.join(dir_path, file)

        return list(self.id_to_file.keys())

    def get_document_metadata(self, document_id):
        file_path = self.id_to_file[document_id]
        metadata = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "size": metadata.st_size,
            "created": metadata.st_ctime,
            "modified": metadata.st_mtime
        }

    def get_document_text(self, document_id):
        file_path = self.id_to_file[document_id]

        with open(file_path, "r") as f:
            return f.read()

    def process_batch(self, batch_ids):
        logging.info(f"Started processing batch of size {len(batch_ids)}")

        batch_metadata = []
        batch_contents = []

        try:
            batch_metadata = [self.get_document_metadata(
                document_id) for document_id in batch_ids]
            batch_contents = [self.get_document_text(
                document_id) for document_id in batch_ids]
        except Exception as e:
            logging.error(f"Exception occurred while processing batch: {e}")

        return batch_ids, batch_metadata, batch_contents
