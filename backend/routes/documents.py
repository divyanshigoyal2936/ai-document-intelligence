"""
M8 - /documents routes

Handles the "document ingestion" side of the app over HTTP:
upload a PDF -> save it -> ingest it (chunks/embeddings/FAISS via
rag_pipeline_langchain from M7) -> remember it so /chat can query it.
"""

import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from schemas.document import DocumentResponse, DocumentListResponse, DeleteResponse
from services.document_store import DOCUMENTS
from services.rag_pipeline_langchain import ingest_pdf, build_rag_chain

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a PDF upload, save it to disk, and run the full ingestion
    pipeline (M7) so it's immediately ready to be questioned.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    document_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    # Read the uploaded file's bytes and write them to disk.
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # This is the same ingest_pdf() from M7 - extract, chunk, embed, index.
    # It's a synchronous, potentially slow call; production versions of
    # this route would run it in the background (M13/M14 territory).
    retriever = ingest_pdf(save_path)
    chain = build_rag_chain(retriever)

    DOCUMENTS[document_id] = {
        "filename": file.filename,
        "path": save_path,
        "chain": chain,
    }

    return DocumentResponse(document_id=document_id, filename=file.filename)


@router.get("", response_model=DocumentListResponse)
def list_documents():
    """Return every document that's been uploaded and ingested so far."""
    documents = [
        DocumentResponse(document_id=doc_id, filename=info["filename"])
        for doc_id, info in DOCUMENTS.items()
    ]
    return DocumentListResponse(documents=documents)


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    """Remove a document from the store (and delete its file from disk)."""
    if document_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = DOCUMENTS[document_id]["path"]
    if os.path.exists(file_path):
        os.remove(file_path)

    del DOCUMENTS[document_id]

    return DeleteResponse(document_id=document_id, deleted=True)
