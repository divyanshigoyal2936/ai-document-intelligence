"""
M7 - LangChain RAG
Concept: Organize the RAG workflow with LangChain

Everything here does the SAME job as M2-M6 combined - PDF -> chunks ->
embeddings -> vector search -> LLM answer. The difference is we now use
LangChain's pre-built pieces instead of the ones we hand-wrote ourselves.

Why bother, if M6 already worked? Two reasons:
1. Less code to maintain for standard steps (loading, splitting,
   vector stores) that LangChain has already solved well.
2. LangChain gives us a common "interface" (Documents, Retrievers,
   Chains) that plugs into a huge ecosystem - this becomes important
   once we add things like streaming, memory, or agents later.

It's worth understanding both versions: M6 shows you exactly what's
happening under the hood (so nothing feels like "magic"), and M7 shows
you the more productive way to build it once you understand the pieces.

Mapping from our M2-M6 code to LangChain's equivalents:
    pdf_processor.py (M2)     -> PyPDFLoader
    text_splitter.py (M3)     -> RecursiveCharacterTextSplitter
    embedding_service.py (M4) -> OpenAIEmbeddings
    vector_store.py (M5)      -> FAISS (langchain_community.vectorstores)
    rag_pipeline.py (M6)      -> create_retrieval_chain
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

if not api_key or api_key == "your_openai_api_key_here":
    raise ValueError(
        "LLM_API_KEY is missing. Open backend/.env and paste your real "
        "OpenAI API key in place of the placeholder."
    )

# LangChain's OpenAI classes read OPENAI_API_KEY from the environment by
# default, so we set it explicitly here to keep using our LLM_API_KEY name.
os.environ["OPENAI_API_KEY"] = api_key

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# The same "answer only from context" instruction as M6's build_prompt(),
# now expressed as a LangChain prompt template. {context} is filled in
# automatically by create_stuff_documents_chain with the retrieved chunks.
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
    # PyPDFLoader automatically gives each page its own Document object
    # with page_content (the text) and metadata (including "page") -
    # this replaces our manual pdf_processor.py page-number bookkeeping.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # RecursiveCharacterTextSplitter is a smarter version of our
    # word-count chunker from M3: it tries to split on paragraph/sentence
    # boundaries first, only falling back to hard cuts when necessary.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = splitter.split_documents(documents)

    print(f"Ingesting {len(split_documents)} chunks from {len(documents)} pages... "
          f"(this calls the embeddings API once per chunk)")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # FAISS.from_documents does what our build_index() did manually:
    # embeds every chunk and stores the vectors + text/metadata together.
    vectorstore = FAISS.from_documents(split_documents, embeddings)

    # as_retriever() wraps the vector store as a "retriever" - a standard
    # LangChain interface that just takes a question and returns Documents.
    # k=3 mirrors the top_k=3 we used with our own search() in M5.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


def build_rag_chain(retriever):
    """
    Wire the retriever and the LLM together into one chain:
    question -> retriever -> context -> LLM -> answer

    This replaces our hand-written ask_question() from M6.
    """
    llm = ChatOpenAI(model=CHAT_MODEL)

    # create_stuff_documents_chain "stuffs" all retrieved Documents into
    # the {context} slot of our prompt, then calls the LLM.
    combine_docs_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

    # create_retrieval_chain wires it all together: given {"input": question},
    # it first calls the retriever, then feeds the results into combine_docs_chain.
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain


def ask_question(rag_chain, question: str) -> dict:
    """
    Run a question through the chain and return the answer plus source pages,
    in the same shape as M6's ask_question() for an easy comparison.
    """
    result = rag_chain.invoke({"input": question})

    # result["context"] is the list of retrieved Documents; each one's
    # metadata["page"] is the 0-indexed page number PyPDFLoader assigned.
    sources = sorted({doc.metadata.get("page", -1) + 1 for doc in result["context"]})

    return {"answer": result["answer"], "sources": sources}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline_langchain.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=== AI Document Intelligence - M7: LangChain RAG ===\n")
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
