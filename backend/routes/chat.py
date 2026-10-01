"""
M8 - /chat routes

Handles the "question answering" side of the app over HTTP: given a
document_id and a question, run it through that document's RAG chain
(from M7) and return the grounded answer plus source pages.
"""

from fastapi import APIRouter, HTTPException

from schemas.chat import ChatRequest, ChatResponse
from services.document_store import DOCUMENTS
from services.rag_pipeline_langchain import ask_question

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest):
    """Ask a question about a previously uploaded document."""
    document = DOCUMENTS.get(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload it first via POST /documents/upload.",
        )

    result = ask_question(document["chain"], request.question)

    return ChatResponse(answer=result["answer"], sources=result["sources"])
