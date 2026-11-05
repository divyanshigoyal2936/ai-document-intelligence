"""
M8 - FastAPI
Purpose: Expose document and chat operations as APIs

All it does is create the FastAPI app and plug in the two routers:
- routes/documents.py -> upload/list/delete PDFs
- routes/chat.py      -> ask questions about an uploaded PDF

Run it with:
    uvicorn main:app --reload
Then open http://localhost:8000/docs for interactive API docs.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import documents, chat

app = FastAPI(
    title="AI Document Intelligence API",
    description="Upload PDFs and ask questions about them using RAG.",
    version="0.1.0",
)

# Allowed frontend origins come from the FRONTEND_ORIGINS environment variable
# (comma separated). It defaults to "*" (everyone) until a frontend exists;
# once the Next.js app is deployed, set it to that exact URL.
frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "AI Document Intelligence API is running."}