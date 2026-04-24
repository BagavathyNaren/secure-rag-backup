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
)

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

        async def stream_tokens() -> AsyncGenerator[str, None]:
            """
            Yield tokens as they arrive from the LLM.
            Also run final guardrails after completion.
            """
            full_answer = ""

            # Stream each token
            async for chunk in rag_chain.astream(user_input):
                full_answer += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            # ---- Output Guardrails (post-stream) ----
            try:
                full_answer = model_guard_check(full_answer)
                full_answer = enforce_one_sentence(full_answer)
                confidence = compute_confidence(retrieved_docs, full_answer)

                log_event("ANSWER", full_answer)

                # Final message with metadata
                yield f"""data: {json.dumps({
                         'done': True,
                        'answer': full_answer,
                        'confidence': confidence
                    })}\n\n"""

            except ValueError as ve:
                yield f"data: {json.dumps({'error': str(ve)})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': 'Internal guardrail error'})}\n\n"

        return StreamingResponse(
            stream_tokens(),
            media_type="text/event-stream"
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


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
# RUN (local dev)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)