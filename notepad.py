BATCH_SIZE = 250
num_batches = len(document_ids) // BATCH_SIZE

# Download first batch of documents and train initial model
initial_ids = document_ids[0:BATCH_SIZE]
token = get_access_token(username, password)
try:
    print(f"Started processing batch 1/{num_batches}")    
    initial_docs = [get_document_text(document_id, token) for document_id in document_ids[0:BATCH_SIZE]]
    model = Top2Vec(documents=initial_docs, document_ids=initial_ids, speed="learn", workers=8, ngram_vocab=True)
    print(f"Finished processing batch 1/{num_batches}")
except Exception as e:
    print(e)

# Loop over remaining batches, downloaded documents and add them to the model
for i in range(1, num_batches):
    start = i * BATCH_SIZE
    end = (i + 1) * BATCH_SIZE
    batch_ids = document_ids[start:end]
    token = get_access_token(username, password)
    try:
        print(f"Started processing batch {i + 1}/{num_batches}")    
        batch_documents = [get_document_text(document_id, token) for document_id in batch_ids]
        model.add_documents(documents=batch_documents, doc_ids=batch_ids)
        print(f"Finished processing batch {i + 1}/{num_batches}")
    except Exception as e:
        print(e)

# If there are any remaining documents, download them and add them to the model
if len(document_ids) % BATCH_SIZE != 0:
    remaining_ids = document_ids[num_batches * BATCH_SIZE]
    token = get_access_token(username, password)
    try:
        print(f"Started processing remaining documents")
        remaining_documents = [get_document_text(document_id) for document_id in remaining_ids]
        model.add_documents(documents=remaining_documents, doc_ids=remaining_ids)
        print("Finished processing all remaining documents")
    except Exception as e:
        print(e)

# Save trained model
model.save("contracts_model_doc2vec")
