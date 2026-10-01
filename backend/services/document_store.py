"""
M8 helper - Document store

A simple in-memory dictionary that remembers which documents have been
uploaded and ingested, and holds the ready-to-query RAG chain for each
one. This is deliberately NOT a database - that's M10's job. For now
it's just enough state to let documents.py and chat.py routes talk to
the same data.

IMPORTANT: because this lives in memory, it resets every time the
server restarts, and it isn't shared across multiple server processes/
workers. That limitation is exactly what MongoDB (M10) will fix.
"""

# document_id (str) -> {
#     "filename": str,
#     "path": str,
#     "chain": <LangChain RAG chain from rag_pipeline_langchain.build_rag_chain>,
# }
DOCUMENTS: dict[str, dict] = {}
