"""
M5 - FAISS
Pipeline: Vectors -> similarity search

FAISS (Facebook AI Similarity Search) is a library built for exactly one
job: given a query vector, quickly find which of a large set of stored
vectors are numerically closest to it. "Closest" vectors = "most similar
meaning" text, because that's how embeddings work (see M4).

Important limitation to understand: FAISS only stores raw numbers - it
has no idea what "page_number" or "text" mean. So alongside the FAISS
index, we keep a plain Python list called `metadata`, in the exact same
order as the vectors were added. When FAISS tells us "vector at position
7 is a close match," we look up metadata[7] to get back the actual chunk
text and page number.
"""

import faiss
import numpy as np


# All OpenAI text-embedding-3-small vectors have this many dimensions.
EMBEDDING_DIMENSIONS = 1536


def build_index(embedded_chunks: list[dict]):
    """
    Build a FAISS index from a list of chunks that already have an
    "embedding" field (i.e. the output of embedding_service.embed_chunks()).

    Returns:
        (index, metadata)
        index    - the FAISS index, ready to be searched
        metadata - a list of chunk info (chunk_id, page_number, text),
                   in the same order as the vectors inside the index
    """
    # IndexFlatL2 = the simplest FAISS index: it compares the query
    # against every stored vector using straight-line (L2) distance.
    # Slower than fancier index types on huge datasets, but exact and
    # easy to reason about - perfect while learning.
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSIONS)

    vectors = np.array(
        [chunk["embedding"] for chunk in embedded_chunks], dtype="float32"
    )
    index.add(vectors)

    # Keep everything except the embedding itself in metadata - we don't
    # need the raw vector again once it's inside the index.
    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
        }
        for chunk in embedded_chunks
    ]

    return index, metadata


def save_index(index, metadata: list[dict], index_path: str, metadata_path: str):
    """Persist the FAISS index and its metadata to disk."""
    import json

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def load_index(index_path: str, metadata_path: str):
    """Load a previously saved FAISS index and its metadata back into memory."""
    import json

    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return index, metadata


def search(index, metadata: list[dict], query_vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Given a query vector, return the top_k most similar chunks.

    Returns a list of chunk dicts (from metadata) with an added
    "distance" field - smaller distance = more similar.
    """
    query = np.array([query_vector], dtype="float32")

    # distances: how far each match is from the query (smaller = closer)
    # indices: the position of each match inside the index/metadata list
    distances, indices = index.search(query, top_k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:  # FAISS returns -1 if there aren't enough vectors to fill top_k
            continue
        match = {**metadata[idx], "distance": float(distance)}
        results.append(match)

    return results


if __name__ == "__main__":
    import sys
    from pdf_processor import extract_text_from_pdf
    from text_splitter import chunk_pages
    from embedding_service import embed_chunks, get_embedding

    if len(sys.argv) < 3:
        print('Usage: python vector_store.py <path_to_pdf> "<your question>"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    # Again, only embedding a handful of chunks here to keep the test cheap.
    sample_chunks = chunks[:10]
    embedded = embed_chunks(sample_chunks)

    index, metadata = build_index(embedded)
    print(f"Built FAISS index with {index.ntotal} vectors.\n")

    query_vector = get_embedding(question)
    results = search(index, metadata, query_vector, top_k=3)

    print(f"Top {len(results)} matches for: \"{question}\"\n")
    for i, result in enumerate(results, start=1):
        preview = result["text"][:150].replace("\n", " ")
        print(f"{i}. (page {result['page_number']}, distance {result['distance']:.4f})")
        print(f"   {preview}{'...' if len(result['text']) > 150 else ''}\n")
