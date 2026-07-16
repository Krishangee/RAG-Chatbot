"""
app.py
Streamlit chat UI for querying the documents ingested by ingest.py.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

PERSIST_DIR = "chroma_db"
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"
RETRIEVAL_K = 6

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using only the
context provided below, which was retrieved from the user's own documents.

If the answer isn't in the context, say you don't have enough information —
don't make anything up.

Context:
{context}

Question: {question}

Answer:"""


@st.cache_resource
def load_qa_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def main():
    st.set_page_config(page_title="Document Chatbot", page_icon="📄")
    st.title("📄 Chat with your documents")
    st.caption("Retrieval-augmented chatbot powered by Gemini + ChromaDB")

    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set. Add it to your .env file.")
        st.stop()

    if not os.path.isdir(PERSIST_DIR):
        st.warning("No vector store found. Run `python ingest.py` first.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = load_qa_chain()
                result = qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for i, doc in enumerate(sources, start=1):
                            page = doc.metadata.get("page", "?")
                            source = doc.metadata.get("source", "unknown")
                            st.markdown(f"**{i}.** `{source}` — page {page}")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
