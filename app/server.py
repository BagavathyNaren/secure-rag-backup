# app/server.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.extension import _rate_limit_exceeded_handler

from app.secure_rag import secure_rag_invoke

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
# 4️⃣ SECURE RAG ENDPOINT
# ============================================================

@app.post("/secure-rag/invoke", response_model=SecureRAGResponse)
@limiter.limit("10/minute")
async def secure_rag_endpoint(request: Request, body: SecureRAGRequest):

    try:
        result = secure_rag_invoke(
            user_input=body.question,
            user_role=body.role
        )

        return SecureRAGResponse(
            answer=result["answer"],
            confidence=result["confidence"]
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