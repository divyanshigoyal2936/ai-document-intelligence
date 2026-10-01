"""
Request/response shapes for the /chat endpoints.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    document_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[int]  # page numbers the answer was grounded in
