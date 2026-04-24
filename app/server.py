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
    enforce_one_sentence,
    compute_confidence,
    log_event,
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
# 2️⃣ RATE LIMITING (Prevents abuse)
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
    role: str = "employee"  # default role


class SecureRAGResponse(BaseModel):
    answer: str
    confidence: str


# ============================================================
# 4️⃣ SECURE RAG ENDPOINT (STREAMING)
# ============================================================

@app.post("/secure-rag/invoke")
@limiter.limit("10/minute")
async def secure_rag_endpoint(request: Request, body: SecureRAGRequest):
    try:
        # ---- Input Guardrails ----
        detect_prompt_injection(body.question)
        user_input = redact_pii(body.question)

        # ---- Secure Retrieval ----
        retriever = build_secure_retriever(body.role)
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

            # Output guardrails
            try:
                full_answer = model_guard_check(full_answer)
                full_answer = enforce_one_sentence(full_answer)
                confidence = compute_confidence(retrieved_docs, full_answer)
                log_event("ANSWER", full_answer)
                yield f"data: {json.dumps({'done': True, 'answer': full_answer, 'confidence': confidence})}\n\n"
            except ValueError as ve:
                yield f"data: {json.dumps({'error': str(ve)})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    except Exception as e:
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
    """
    Returns detailed health of all components.
    """
    health_status = {
        "status": "ok",
        "service": "Tech Secure RAG",
        "version": "2.0.0",
        "components": {}
    }

    # 1. FAISS index check
    try:
        if _vectorstore is not None:
            # Quick test: perform a dummy similarity search
            _ = _vectorstore.similarity_search("test", k=1)
            health_status["components"]["faiss"] = "healthy"
        else:
            health_status["components"]["faiss"] = "not_loaded"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["faiss"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # 2. OpenAI embeddings check
    try:
        _embeddings.embed_query("health check")
        health_status["components"]["embeddings"] = "healthy"
    except Exception as e:
        health_status["components"]["embeddings"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # 3. LLM check (minimal)
    try:
        _llm.invoke("ping")
        health_status["components"]["llm"] = "healthy"
    except Exception as e:
        health_status["components"]["llm"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # 4. Overall status
    if all(v == "healthy" for v in health_status["components"].values()):
        health_status["status"] = "healthy"

    return health_status

# ============================================================
# RUN (local dev)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)