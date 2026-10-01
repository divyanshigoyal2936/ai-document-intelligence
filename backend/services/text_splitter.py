"""
M3 - Chunking
Pipeline: Text -> smaller overlapping chunks

Why chunk at all? LLMs and embedding models work best on small,
focused pieces of text - not a 50-page book in one go. Chunking also
means that when we later search for "relevant" text (M5 - FAISS), we
get back a tightly focused passage instead of an entire page of mixed
topics.

Why overlapping? If we cut chunks with no overlap, a sentence that
explains something important might get split right down the middle,
with half the meaning in one chunk and half in the next. A small
overlap (e.g. the last 50 words of chunk N also appear at the start
of chunk N+1) makes it much less likely we lose context at the edges.

We chunk each page separately (rather than gluing the whole PDF into
one string first) so every chunk can still remember which page it
came from - that's the metadata we preserved back in M2, and it's
what powers citations in M12.
"""


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 40) -> list[str]:
    """
    Split a single string of text into overlapping chunks, measured in words.

    chunk_size: how many words go in each chunk.
    chunk_overlap: how many words from the end of one chunk are repeated
                   at the start of the next chunk.
    """
    words = text.split()

    if not words:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        # Move the window forward, but step back by chunk_overlap words
        # so the next chunk repeats the tail of this one.
        start = end - chunk_overlap

    return chunks


def chunk_pages(
    pages: list[dict], chunk_size: int = 200, chunk_overlap: int = 40
) -> list[dict]:
    """
    Take the page list produced by pdf_processor.extract_text_from_pdf()
    and turn it into a flat list of chunks, each still tagged with the
    page number it came from.

    Returns:
        [
            {"chunk_id": 0, "page_number": 1, "text": "..."},
            {"chunk_id": 1, "page_number": 1, "text": "..."},
            {"chunk_id": 2, "page_number": 2, "text": "..."},
            ...
        ]
    """
    all_chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)

        for chunk in page_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "page_number": page["page_number"],
                "text": chunk,
            })
            chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    import sys
    from pdf_processor import extract_text_from_pdf

    if len(sys.argv) < 2:
        print("Usage: python text_splitter.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Produced {len(chunks)} chunks from {len(pages)} pages.\n")

    for chunk in chunks[:5]:  # just preview the first 5
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"--- Chunk {chunk['chunk_id']} (page {chunk['page_number']}) ---")
        print(f"{preview}{'...' if len(chunk['text']) > 150 else ''}\n")
