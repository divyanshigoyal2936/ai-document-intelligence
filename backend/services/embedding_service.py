"""
M4 - Embeddings
Pipeline: Chunks -> vectors

An embedding is just a list of numbers (a vector) that represents the
*meaning* of a piece of text. Texts with similar meaning end up with
vectors that are numerically close to each other. That's the property
we'll exploit in M5 (FAISS) to find "chunks that are relevant to this
question" - we're really finding "vectors that are close to the
question's vector."

We use OpenAI's embedding model here, same as we used OpenAI for the
CLI in M1 - one API key, two different endpoints (chat vs embeddings).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

if not api_key or api_key == "your_openai_api_key_here":
    raise ValueError(
        "LLM_API_KEY is missing. Open backend/.env and paste your real "
        "OpenAI API key in place of the placeholder."
    )

client = OpenAI(api_key=api_key)

# text-embedding-3-small is cheap and fast - a good default while learning.
# It outputs vectors of 1536 numbers per piece of text.
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> list[float]:
    """Convert a single piece of text into its embedding vector."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Take the chunk list produced by text_splitter.chunk_pages() and add
    an "embedding" field to each chunk, containing its vector.

    Input chunk:  {"chunk_id": 0, "page_number": 1, "text": "..."}
    Output chunk: {"chunk_id": 0, "page_number": 1, "text": "...", "embedding": [0.01, -0.02, ...]}
    """
    embedded_chunks = []

    for chunk in chunks:
        vector = get_embedding(chunk["text"])
        embedded_chunks.append({**chunk, "embedding": vector})

    return embedded_chunks


if __name__ == "__main__":
    import sys
    from pdf_processor import extract_text_from_pdf
    from text_splitter import chunk_pages

    if len(sys.argv) < 2:
        print("Usage: python embedding_service.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    # Only embed the first 3 chunks here as a cheap smoke test -
    # embedding a whole book costs real API calls/money.
    sample_chunks = chunks[:3]
    embedded = embed_chunks(sample_chunks)

    print(f"Embedded {len(embedded)} sample chunks out of {len(chunks)} total.\n")

    for chunk in embedded:
        vector_preview = chunk["embedding"][:5]
        print(f"--- Chunk {chunk['chunk_id']} (page {chunk['page_number']}) ---")
        print(f"Vector length: {len(chunk['embedding'])}")
        print(f"First 5 values: {vector_preview}\n")
