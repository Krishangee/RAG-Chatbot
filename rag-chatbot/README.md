# 📄 RAG Chatbot — Chat With Your Documents

A Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about your own PDF documents. It ingests PDFs, splits them into chunks, embeds them with Google Gemini, stores the vectors locally in ChromaDB, and answers questions through a Streamlit chat interface — grounded only in the retrieved context.

## How it works

1. **Ingest (`ingest.py`)** — loads PDFs from `data/`, splits them into overlapping chunks (1000 chars, 150 overlap), embeds each chunk with Gemini's embedding model, and persists them in a local ChromaDB store.
2. **Chat (`app.py`)** — on each question, retrieves the top 6 most relevant chunks from ChromaDB and passes them to Gemini as context, so answers are grounded in your documents instead of the model's general knowledge.

## Project structure

```
rag-chatbot/
├── data/              # Drop your PDFs here
├── chroma_db/          # Generated vector store (git-ignored)
├── ingest.py            # Chunking + embedding pipeline
├── app.py               # Streamlit chat app
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## Usage

1. Drop one or more PDFs into `data/`.
2. Build the vector store:
   ```bash
   python ingest.py
   ```
3. Launch the chatbot:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501` and start asking questions.

## Tech stack

Python, LangChain, Google Gemini (embeddings + chat), ChromaDB, Streamlit
