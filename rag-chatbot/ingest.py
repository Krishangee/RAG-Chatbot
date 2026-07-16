"""
ingest.py
Reads every PDF in data/, splits it into overlapping chunks, embeds each
chunk with Google's Gemini embedding model, and persists the vectors in a
local ChromaDB store so app.py can query them later.

Run this once whenever you add/change documents in data/:
    python ingest.py
"""

import os
import time
import glob

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "models/embedding-001"

# Gemini's free tier rate-limits embedding calls, so we batch + sleep
# between requests instead of firing them all at once.
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 2  # seconds


def load_documents():
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in '{DATA_DIR}/'. Drop some in there first."
        )

    docs = []
    for path in pdf_paths:
        print(f"Loading {path} ...")
        docs.extend(PyPDFLoader(path).load())
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} pages into {len(chunks)} chunks.")
    return chunks


def embed_and_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vector_store.add_documents(batch)
        done = min(start + BATCH_SIZE, total)
        print(f"Embedded {done}/{total} chunks")

        if done < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Done. Vector store persisted at '{PERSIST_DIR}/'.")


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to a .env file in the project root."
        )

    docs = load_documents()
    chunks = chunk_documents(docs)
    embed_and_store(chunks)


if __name__ == "__main__":
    main()
