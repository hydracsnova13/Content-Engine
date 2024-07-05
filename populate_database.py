import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CHROMA_PATH = "chroma"
DATA_PATH = "data"
BATCH_SIZE = 50  # Adjust batch size as needed
SLEEP_INTERVAL = 2  # Sleep for 2 seconds between batches
NUM_WORKERS = 4  # Number of parallel workers

logging.basicConfig(level=logging.INFO)

def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        logging.info("✨ Clearing Database")
        clear_database()
        reset_checkpoint()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    logging.info("Loading documents...")
    documents = document_loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: list[Document]):
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    logging.info("Loading existing database...")
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logging.info(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    checkpoint = load_checkpoint()
    start_index = checkpoint.get('last_index', 0)

    if new_chunks:
        logging.info(f"👉 Adding new documents: {len(new_chunks)}")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(add_batch_to_chroma, db, new_chunks[i:i + BATCH_SIZE], i//BATCH_SIZE + 1)
                for i in range(start_index, len(new_chunks), BATCH_SIZE)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                    save_checkpoint(future.result())
                    time.sleep(SLEEP_INTERVAL)
                except Exception as e:
                    logging.error(f"Error adding batch: {e}")
    else:
        logging.info("✅ No new documents to add")

def add_batch_to_chroma(db, batch, batch_number):
    new_chunk_ids = [chunk.metadata["id"] for chunk in batch]
    logging.info(f"Adding batch {batch_number} with {len(batch)} documents...")
    db.add_documents(batch, ids=new_chunk_ids)
    db.persist()
    logging.info(f"Batch {batch_number} added.")
    return batch_number * BATCH_SIZE

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def save_checkpoint(last_index):
    checkpoint = {'last_index': last_index}
    with open('checkpoint.json', 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint():
    if os.path.exists('checkpoint.json'):
        with open('checkpoint.json', 'r') as f:
            return json.load(f)
    return {}

def reset_checkpoint():
    if os.path.exists('checkpoint.json'):
        os.remove('checkpoint.json')

if __name__ == "__main__":
    main()
