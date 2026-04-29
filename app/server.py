# app/server.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.extension import _rate_limit_exceeded_handler
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import traceback
import logging
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.secure_rag import (
    detect_prompt_injection,
    redact_pii,
    build_secure_retriever,
    secure_prompt,
    _llm,
    model_guard_check,
    compute_confidence,
    log_event,
    audit,              # ← Import audit logger
    new_trace_id,       # ← Import trace ID generator
    _vectorstore,
    _embeddings
)

logger = logging.getLogger(__name__)

# ============================================================
# 1️⃣ FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Tech Secure RAG API",
    description="Enterprise-secured RAG with guardrails + RBAC",
    version="2.0.0"
)

# ============================================================
# 2️⃣ RATE LIMITING
# ============================================================

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ============================================================
# 3️⃣ REQUEST MODEL
# ============================================================

class SecureRAGRequest(BaseModel):
    question: str
    role: str = "employee"


class SecureRAGResponse(BaseModel):
    answer: str
    confidence: str


# ============================================================
# 4️⃣ SECURE RAG ENDPOINT (STREAMING + TRACE ID)
# ============================================================

@app.post("/secure-rag/invoke")
@limiter.limit("10/minute")
async def secure_rag_endpoint(request: Request, body: SecureRAGRequest):

    # ✅ Generate ONE unique trace_id for this entire request
    trace_id = new_trace_id()

    audit.log("REQUEST_START", trace_id, {
        "role": body.role,
        "question_preview": body.question[:80]
    })

    try:
        # ---- Input Guardrails ----
        detect_prompt_injection(body.question, trace_id)
        user_input = redact_pii(body.question, trace_id)

        # ---- Secure Retrieval ----
        retriever = build_secure_retriever(body.role, trace_id)
        retrieved_docs = retriever(user_input)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # ---- Streaming Chain ----
        setup = RunnableParallel(
            context=lambda _: context,
            question=RunnablePassthrough()
        )

        rag_chain = setup | secure_prompt | _llm | StrOutputParser()

        async def stream_tokens():
            full_answer = ""

            async for chunk in rag_chain.astream(user_input):
                full_answer += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            # ---- Output Guardrails ----
            try:
                full_answer = model_guard_check(full_answer, context, trace_id)
                confidence = compute_confidence(retrieved_docs, full_answer)

                # ✅ Log with real trace_id (not "system")
                audit.log("ANSWER", trace_id, {
                    "message": full_answer[:300],
                    "confidence": confidence,
                    "role": body.role,
                    "sources": [d.metadata.get("file_name") for d in retrieved_docs]
                })

                yield f"data: {json.dumps({'done': True, 'answer': full_answer, 'confidence': confidence})}\n\n"

            except ValueError as ve:
                audit.log("REQUEST_FAILED", trace_id, {
                    "error": str(ve),
                    "role": body.role
                })
                yield f"data: {json.dumps({'error': str(ve)})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    except Exception as e:
        audit.log("REQUEST_FAILED", trace_id, {
            "error": str(e),
            "role": body.role
        })
        logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# ============================================================
# 5️⃣ HEALTH CHECK
# ============================================================

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Tech Secure RAG",
        "version": "2.0.0",
        "endpoint": "/secure-rag/invoke"
    }


# ============================================================
# 6️⃣ DETAILED HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    health_status = {
        "status": "ok",
        "service": "Tech Secure RAG",
        "version": "2.0.0",
        "components": {}
    }

    # FAISS check
    try:
        if _vectorstore is not None:
            _ = _vectorstore.similarity_search("test", k=1)
            health_status["components"]["faiss"] = "healthy"
        else:
            health_status["components"]["faiss"] = "not_loaded"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["faiss"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Embeddings check
    try:
        _embeddings.embed_query("health check")
        health_status["components"]["embeddings"] = "healthy"
    except Exception as e:
        health_status["components"]["embeddings"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # LLM check
    try:
        _llm.invoke("ping")
        health_status["components"]["llm"] = "healthy"
    except Exception as e:
        health_status["components"]["llm"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Overall
    if all(v == "healthy" for v in health_status["components"].values()):
        health_status["status"] = "healthy"

    return health_status


# ============================================================
# RUN (local dev)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)