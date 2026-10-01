"""
M6 - Basic RAG
Pipeline: Question -> retrieval -> context -> LLM -> answer

This is the milestone where the project actually becomes "RAG"
(Retrieval-Augmented Generation). Every piece built so far gets
wired together into one flow:

    1. Ingest the PDF once  -> pages -> chunks -> embeddings -> FAISS index
    2. For each question    -> embed question -> search index -> get top chunks
    3. Build a prompt that says "using ONLY this context, answer this question"
    4. Send that prompt to the LLM
    5. Return the answer (and which pages it came from)

Why "using ONLY this context"? Without that instruction, the LLM will
happily answer from its own general training knowledge, which defeats
the purpose of RAG - we specifically want answers grounded in *this*
document, not the model's guesses.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from pdf_processor import extract_text_from_pdf
from text_splitter import chunk_pages
from embedding_service import embed_chunks, get_embedding
from vector_store import build_index, search

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

if not api_key or api_key == "your_openai_api_key_here":
    raise ValueError(
        "LLM_API_KEY is missing. Open backend/.env and paste your real "
        "OpenAI API key in place of the placeholder."
    )

client = OpenAI(api_key=api_key)
CHAT_MODEL = "gpt-4o-mini"


def ingest_pdf(pdf_path: str):
    """
    Run the full ingestion pipeline once for a PDF:
    PDF -> pages -> chunks -> embeddings -> FAISS index

    Returns (index, metadata) - keep these in memory (or save them,
    see vector_store.save_index) so you don't have to re-embed the
    whole document on every single question.
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Ingesting {len(chunks)} chunks from {len(pages)} pages... "
          f"(this calls the embeddings API once per chunk)")
    embedded_chunks = embed_chunks(chunks)

    index, metadata = build_index(embedded_chunks)
    return index, metadata


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Combine the retrieved chunks and the user's question into a single
    prompt string for the LLM. Each chunk is labeled with its page
    number so the model can naturally refer to "page 3" etc. in its
    answer if it wants to.
    """
    context_sections = []
    for chunk in retrieved_chunks:
        context_sections.append(f"[Page {chunk['page_number']}]\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_sections)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't know based on the provided document."

Context:
{context_text}

Question: {question}

Answer:"""

    return prompt


def ask_question(index, metadata: list[dict], question: str, top_k: int = 3) -> dict:
    """
    Run the question-answering half of the pipeline against an already
    built index: question -> retrieval -> context -> LLM -> answer.

    Returns:
        {
            "answer": "...",
            "sources": [1, 3]  # unique page numbers the answer drew from
        }
    """
    query_vector = get_embedding(question)
    retrieved_chunks = search(index, metadata, query_vector, top_k=top_k)

    prompt = build_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about a document."},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content
    sources = sorted({chunk["page_number"] for chunk in retrieved_chunks})

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=== AI Document Intelligence - M6: Basic RAG ===\n")
    index, metadata = ingest_pdf(pdf_path)
    print(f"Ingestion complete. Index has {index.ntotal} vectors.\n")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        result = ask_question(index, metadata, question)
        print(f"AI: {result['answer']}")
        print(f"Sources: page(s) {result['sources']}\n")
