"""
M8 - FastAPI
Purpose: Expose document and chat operations as APIs

This file replaces the CLI loop from M1 (moved to cli_m1.py) as the
project's entry point. Instead of one Python process you type questions
into, we now run a web server that a frontend (M9 - Next.js) or any
HTTP client (curl, Postman, etc.) can talk to.

All it does is create the FastAPI app and plug in the two routers:
- routes/documents.py -> upload/list/delete PDFs
- routes/chat.py      -> ask questions about an uploaded PDF

Run it with:
    uvicorn main:app --reload
Then open http://localhost:8000/docs for interactive API docs that
FastAPI generates automatically from our Pydantic schemas.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import documents, chat

app = FastAPI(
    title="AI Document Intelligence API",
    description="Upload PDFs and ask questions about them using RAG.",
    version="0.1.0",
)

# Allow the Next.js frontend (M9), running on a different port/origin,
# to call this API from the browser. Wide open for local development -
# this should be locked down to specific origins before deployment (M15).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "AI Document Intelligence API is running."}
