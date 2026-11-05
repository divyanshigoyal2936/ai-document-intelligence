"""
M7 - LangChain RAG (free stack)
Concept: Organize the RAG workflow with LangChain

Same pipeline as before: PDF -> chunks -> embeddings -> FAISS -> LLM answer.
The only change is WHICH providers we plug in, so everything runs for free:

    Embeddings: OpenAIEmbeddings  ->  FastEmbedEmbeddings (runs locally, no API, no cost)
    Chat model: OpenAI gpt-4o-mini ->  Groq (free tier) through its OpenAI-compatible API

Mapping from our M2-M6 code to LangChain's equivalents:
    pdf_processor.py (M2)     -> PyPDFLoader
    text_splitter.py (M3)     -> RecursiveCharacterTextSplitter
    embedding_service.py (M4) -> FastEmbedEmbeddings
    vector_store.py (M5)      -> FAISS (langchain_community.vectorstores)
    rag_pipeline.py (M6)      -> create_retrieval_chain
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Add GROQ_API_KEY=your_key to backend/.env "
        "(locally) or to the environment variables on your host."
    )

# Groq exposes an OpenAI-compatible endpoint, so ChatOpenAI works with it.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-120b"

# Small English embedding model, runs on CPU with ONNX (about 130 MB).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Created once and reused, so the model is loaded only one time.
embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {input}

Answer:"""
)


def ingest_pdf(pdf_path: str):
    """
    Run the full ingestion pipeline using LangChain components:
    PDF -> Documents -> split Documents -> embeddings -> FAISS vector store

    Returns a LangChain retriever, ready to be plugged into a chain.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = splitter.split_documents(documents)

    print(f"Ingesting {len(split_documents)} chunks from {len(documents)} pages...")

    vectorstore = FAISS.from_documents(split_documents, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


def build_rag_chain(retriever):
    """
    Wire the retriever and the LLM together into one chain:
    question -> retriever -> context -> LLM -> answer
    """
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=groq_api_key,
        base_url=GROQ_BASE_URL,
        temperature=0,
    )

    combine_docs_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain


def ask_question(rag_chain, question: str) -> dict:
    """
    Run a question through the chain and return the answer plus source pages.
    """
    result = rag_chain.invoke({"input": question})

    # Each retrieved Document's metadata["page"] is the 0-indexed page number.
    sources = sorted({doc.metadata.get("page", -1) + 1 for doc in result["context"]})

    return {"answer": result["answer"], "sources": sources}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline_langchain.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=== AI Document Intelligence - M7: LangChain RAG (free stack) ===\n")
    retriever = ingest_pdf(pdf_path)
    rag_chain = build_rag_chain(retriever)
    print("Ingestion complete.\n")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        result = ask_question(rag_chain, question)
        print(f"AI: {result['answer']}")
        print(f"Sources: page(s) {result['sources']}\n")