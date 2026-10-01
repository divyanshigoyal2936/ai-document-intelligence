"""
Request/response shapes for the /documents endpoints.

FastAPI uses these Pydantic models to: validate incoming data, generate
the interactive API docs at /docs automatically, and serialize outgoing
responses consistently.
"""

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """Returned after a successful upload, and as part of the document list."""
    document_id: str
    filename: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
