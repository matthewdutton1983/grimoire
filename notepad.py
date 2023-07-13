BATCH_SIZE = 250
num_batches = len(unique_ids) // BATCH_SIZE

document_ids = []
document_names = []
document_contents = []

def process_batch(batch_ids, batch_num):
    token = get_access_token(username, password)
    logging.info(f"Started processing batch {batch_num}/{num_batches}")

    try:
        batch_names = [get_document_name(document_id, token) for document_id in batch_ids]
        batch_contents = [get_document_text(document_id, token) for document_id in batch_ids]
        
        document_ids.append(batch_ids)
        document_names.append(batch_names)
        document_contents.append(batch_contents)
    except Exception as e:
        logging.error(f"Exception occurred while downloading batch {batch_num}: {e}")

threads = []
for i in range(num_batches):
    start = i * BATCH_SIZE
    end = (i + 1) * BATCH_SIZE
    batch_ids = unique_ids[start:end]

    t = threading.Thread(target=process_batch, args=(batch_ids, i + 1))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

remaining_ids = unique_ids[num_batches * BATCH_SIZE:]
if remaining_ids:
    logging.info("Starting processing remaining documents")
    token = get_access_token(username, password)
    try:
        remaining_names = [get_document_name(remaining_id, token) for remaining_id in remaining_ids]
        remaining_contents = [get_document_text(remaining_id, token) for remaining_id in remaining_ids]
        
        document_ids.append(remaining_ids)
        document_names.append(remaining_names)
        document_contents.append(remaining_contents)
    except Exception as e:
        logging.error(f"Exception occurred while downloading remaining documents: {e}")
    logging.info("Finished processing all documents")
