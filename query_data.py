# context_retrieval.py

import os
import asyncio
import logging
import re
import nltk
from nltk.corpus import stopwords
from string import punctuation
import tiktoken
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

# Disable telemetry for Chroma
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"

CHROMA_PATH = "chroma"

CHARACTER_LIMIT = 15000  # Adjusted to a higher limit based on token estimation

logging.basicConfig(level=logging.INFO)

# Ensure you have the necessary NLTK data files
nltk.download('stopwords')

# Function to preprocess and condense the text
def preprocess_text(text):
    # Remove punctuation
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    
    # Tokenize the text
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Reconstruct the text
    condensed_text = " ".join(filtered_words)
    
    return condensed_text

# Function to summarize the text using GPT-2
def summarize_text_gpt2(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    inputs = tokenizer.encode("Summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=300, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

async def query_rag(query_text: str, top_k: int = 5):
    logging.info("Initializing the embedding function...")
    embedding_function = get_embedding_function()

    logging.info("Loading the Chroma database...")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    logging.info("Searching for relevant documents...")
    results = db.similarity_search_with_score(query_text, k=top_k)

    logging.info("Relevant documents found:")
    context_parts = []
    for i, (doc, score) in enumerate(results):
        logging.info(f"Document {i+1}:")
        logging.info(f"  ID: {doc.metadata.get('id')}")
        logging.info(f"  Score: {score}")
        logging.info(f"  Content: {doc.page_content[:500]}...")  # Print the first 500 characters
        # Truncate each document section to 200 characters
        truncated_content = doc.page_content[:200]
        context_parts.append(truncated_content)

    context_text = "\n\n---\n\n".join(context_parts)

    # Replace newlines and tabs with spaces
    context_text = context_text.replace("\n", " ").replace("\t", " ")

    # Preprocess and condense the context text
    condensed_context = preprocess_text(context_text)

    # Summarize the context text using GPT-2
    summarized_context = summarize_text_gpt2(condensed_context)

    # Tokenize and count tokens using tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(summarized_context)
    token_count = len(tokens)

    logging.info("Summarized context text:")
    logging.info(summarized_context)
    logging.info(f"Token count: {token_count}")

    return summarized_context

# Function to call from the Streamlit app
def get_condensed_context(query_text: str, top_k: int = 5):
    return asyncio.run(query_rag(query_text, top_k))
