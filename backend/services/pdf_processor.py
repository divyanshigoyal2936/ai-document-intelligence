"""
M2 - PDF Reader
Pipeline: PDF -> extracted text

This module knows how to open a PDF file and pull the raw text out of it,
one page at a time. We keep page numbers attached to the text because
later (M12 - Citations) we'll need to tell the user "this answer came
from page 7" - so it's important to preserve that metadata from the start,
even though we don't use it yet.
"""

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Read a PDF file and return its text, split by page.

    Returns a list of dicts like:
        [
            {"page_number": 1, "text": "..."},
            {"page_number": 2, "text": "..."},
            ...
        ]

    Each dict is one page. We use page_number starting at 1 (not 0)
    because that matches how humans refer to pages in a document.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for index, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""  # extract_text() can return None
        cleaned_text = raw_text.strip()

        pages.append({
            "page_number": index + 1,
            "text": cleaned_text,
        })

    return pages


def extract_full_text(pdf_path: str) -> str:
    """
    Convenience helper: return the entire PDF as one big string,
    useful for quick testing before we introduce chunking in M3.
    """
    pages = extract_text_from_pdf(pdf_path)
    return "\n\n".join(page["text"] for page in pages if page["text"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(pages)} pages from: {pdf_path}\n")

    for page in pages:
        preview = page["text"][:200].replace("\n", " ")
        print(f"--- Page {page['page_number']} ---")
        print(f"{preview}{'...' if len(page['text']) > 200 else ''}\n")
