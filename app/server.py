# app/server.py

from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.extension import _rate_limit_exceeded_handler
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json
import traceback
import logging
import os

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Wrap auth import so we see the REAL error if it fails ──
try:
    from app.auth import (
        authenticate_user,
        create_access_token,
        get_current_user,
        require_role,
        EXPIRE_MINS
    )
    AUTH_AVAILABLE = True
except Exception as e:
    import traceback as tb
    print(f"[STARTUP ERROR] Failed to import app.auth: {e}")
    print(tb.format_exc())
    AUTH_AVAILABLE = False

from app.secure_rag import (
    detect_prompt_injection,
    redact_pii,
    build_secure_retriever,
    secure_prompt,
    _llm,
    model_guard_check,
    compute_confidence,
    scan_context_for_credentials,
    scan_answer_for_sensitive_terms,
    pre_filter_check,
    log_event,
    audit,
    new_trace_id,
    llm_cache,
    _vectorstore,
    _embeddings
)

logger = logging.getLogger(__name__)

# ============================================================
# 1️⃣ FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Tech Secure RAG API",
    description="Enterprise-secured RAG with JWT auth + RBAC + guardrails",
    version="3.0.0"
)

# ============================================================
# 2️⃣ RATE LIMITING
# ============================================================

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ============================================================
# 3️⃣ REQUEST MODELS
# ============================================================

class SecureRAGRequest(BaseModel):
    question: str
    role: str = "employee"    # Kept as fallback if auth unavailable


class SecureRAGResponse(BaseModel):
    answer: str
    confidence: str
    cached: bool = False


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in_minutes: int
    user_id: str
    email: str
    role: str


# ============================================================
# 3b️⃣ EVALUATION MODELS
# ============================================================

class EvalCase(BaseModel):
    id: Optional[str] = None
    question: str
    role: str = "employee"
    should_block: bool = False
    expect_confidence: Optional[str] = Field(default=None)
    expect_answer_contains: List[str] = Field(default_factory=list)
    expect_sources_contains: List[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    cases: List[EvalCase]
    include_answer: bool = Field(default=False)
    include_sources: bool = Field(default=True)


# ============================================================
# EVAL AUTH GUARD
# ============================================================

EVAL_API_KEY = os.getenv("EVAL_API_KEY", "")

def _require_eval_key(x_eval_key: Optional[str]):
    if not EVAL_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Evaluation endpoint is disabled. Set EVAL_API_KEY secret."
        )
    if not x_eval_key or x_eval_key != EVAL_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: invalid or missing x-eval-key header."
        )


# ============================================================
# 4️⃣ AUTH ENDPOINTS (only if auth loaded successfully)
# ============================================================

if AUTH_AVAILABLE:

    @app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
    @limiter.limit("10/minute")
    async def login(
        request: Request,
        form_data: OAuth2PasswordRequestForm = Depends()
    ):
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            audit.log("LOGIN_FAILED", "auth", {
                "username": form_data.username,
                "reason": "invalid_credentials"
            })
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = create_access_token(user)

        audit.log("LOGIN_SUCCESS", "auth", {
            "user_id":  user["user_id"],
            "username": user["username"],
            "email":    user["email"],
            "role":     user["role"]
        })

        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in_minutes=EXPIRE_MINS,
            user_id=user["user_id"],
            email=user["email"],
            role=user["role"]
        )

    @app.get("/auth/me", tags=["Authentication"])
    async def whoami(current_user: dict = Depends(get_current_user)):
        return {
            "user_id": current_user["user_id"],
            "email":   current_user["email"],
            "role":    current_user["role"],
            "sub":     current_user["sub"]
        }

else:
    # Auth failed to load — expose a clear error endpoint
    @app.get("/auth/status", tags=["Authentication"])
    async def auth_status():
        return {
            "status": "AUTH_MODULE_FAILED_TO_LOAD",
            "detail": "Check container logs for the real error."
        }


# ============================================================
# 5️⃣ SECURE RAG ENDPOINT (STREAMING)
# ============================================================

@app.post("/secure-rag/invoke", tags=["RAG"])
@limiter.limit("10/minute")
async def secure_rag_endpoint(
    request: Request,
    body: SecureRAGRequest,
):
    trace_id = new_trace_id()

    # If JWT auth is available → use it, else fall back to body.role
    if AUTH_AVAILABLE:
        from fastapi.security import OAuth2PasswordBearer
        from app.auth import decode_token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization header.")
        token = auth_header.split(" ", 1)[1]
        current_user = decode_token(token)
        user_role  = current_user["role"]
        user_id    = current_user["user_id"]
        user_email = current_user["email"]
    else:
        # Fallback — no auth (dev mode only)
        user_role  = body.role
        user_id    = "anonymous"
        user_email = "anonymous"

    audit.log("REQUEST_START", trace_id, {
        "user_id":          user_id,
        "email":            user_email,
        "role":             user_role,
        "question_preview": body.question[:120]
    })

    try:
        detect_prompt_injection(body.question, trace_id)
        user_input = redact_pii(body.question, trace_id)

        # Check cache
        cached = llm_cache.get(user_role, user_input)
        if cached:
            audit.log("CACHE_HIT", trace_id, {
                "user_id":    user_id,
                "role":       user_role,
                "confidence": cached["confidence"]
            })

            async def stream_cached():
                for word in cached["answer"].split(" "):
                    yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                yield f"data: {json.dumps({'done': True, 'answer': cached['answer'], 'confidence': cached['confidence'], 'cached': True})}\n\n"

            return StreamingResponse(stream_cached(), media_type="text/event-stream")

        audit.log("CACHE_MISS", trace_id, {"user_id": user_id, "role": user_role})

        retriever     = build_secure_retriever(user_role, trace_id)
        retrieved_docs = retriever(user_input)
        context       = "\n\n".join(doc.page_content for doc in retrieved_docs)

        setup     = RunnableParallel(context=lambda _: context, question=RunnablePassthrough())
        rag_chain = setup | secure_prompt | _llm | StrOutputParser()

        async def stream_tokens():
            full_answer = ""

            try:
                scan_context_for_credentials(context, trace_id)
            except ValueError as ve:
                audit.log("REQUEST_FAILED", trace_id, {"error": str(ve), "stage": "context_scan"})
                yield f"data: {json.dumps({'error': str(ve)})}\n\n"
                return

            async for chunk in rag_chain.astream(user_input):
                full_answer += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            try:
                full_answer = pre_filter_check(full_answer, trace_id)
                full_answer = scan_answer_for_sensitive_terms(full_answer, trace_id)
                full_answer = model_guard_check(full_answer, context, trace_id)
                confidence  = compute_confidence(retrieved_docs, full_answer)

                llm_cache.set(user_role, user_input, {
                    "answer":     full_answer,
                    "confidence": confidence
                })

                audit.log("ANSWER", trace_id, {
                    "user_id":    user_id,
                    "email":      user_email,
                    "role":       user_role,
                    "message":    full_answer[:500],
                    "confidence": confidence,
                    "cached":     False,
                    "sources":    [d.metadata.get("file_name") for d in retrieved_docs]
                })

                yield f"data: {json.dumps({'done': True, 'answer': full_answer, 'confidence': confidence, 'cached': False})}\n\n"

            except ValueError as ve:
                audit.log("REQUEST_FAILED", trace_id, {"error": str(ve), "stage": "output_guard"})
                yield f"data: {json.dumps({'error': str(ve)})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    except ValueError as ve:
        audit.log("REQUEST_FAILED", trace_id, {"error": str(ve), "stage": "input_guard"})
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        audit.log("REQUEST_FAILED", trace_id, {"error": str(e), "stage": "unhandled"})
        logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ============================================================
# 6️⃣ EVALUATION ENDPOINT
# ============================================================

@app.post("/secure-rag/eval", tags=["Internal"])
@limiter.limit("5/minute")
async def secure_rag_eval(
    request: Request,
    body: EvalRequest,
    x_eval_key: Optional[str] = Header(default=None, alias="x-eval-key"),
):
    _require_eval_key(x_eval_key)

    batch_id = new_trace_id()
    audit.log("EVAL_BATCH_START", batch_id, {"total_cases": len(body.cases)})

    results = []
    passed  = 0

    for i, case in enumerate(body.cases):
        case_trace_id = new_trace_id()
        case_id       = case.id or f"case_{i + 1}"

        audit.log("EVAL_CASE_START", case_trace_id, {
            "batch_id": batch_id, "case_id": case_id,
            "role": case.role, "should_block": case.should_block,
            "question_preview": case.question[:120]
        })

        answer = None
        confidence = None
        retrieved_sources = []
        blocked   = False
        error_msg = None

        try:
            detect_prompt_injection(case.question, case_trace_id)
            clean_input = redact_pii(case.question, case_trace_id)

            retriever      = build_secure_retriever(case.role, case_trace_id)
            retrieved_docs = retriever(clean_input)
            retrieved_sources = [d.metadata.get("file_name") for d in retrieved_docs]
            context = "\n\n".join(d.page_content for d in retrieved_docs)

            scan_context_for_credentials(context, case_trace_id)

            setup     = RunnableParallel(context=lambda _: context, question=RunnablePassthrough())
            rag_chain = setup | secure_prompt | _llm | StrOutputParser()
            answer    = await rag_chain.ainvoke(clean_input)

            answer     = pre_filter_check(answer, case_trace_id)
            answer     = scan_answer_for_sensitive_terms(answer, case_trace_id)
            answer     = model_guard_check(answer, context, case_trace_id)
            confidence = compute_confidence(retrieved_docs, answer)

        except Exception as e:
            blocked   = True
            error_msg = str(e)
            confidence = "BLOCKED"

        checks = {}
        checks["block_check"] = (blocked == case.should_block)
        checks["confidence_check"] = (confidence == case.expect_confidence) if case.expect_confidence else True

        if case.expect_answer_contains and answer:
            checks["answer_content_check"] = all(kw.lower() in answer.lower() for kw in case.expect_answer_contains)
        elif case.expect_answer_contains and not answer:
            checks["answer_content_check"] = False
        else:
            checks["answer_content_check"] = True

        checks["sources_check"] = all(s in set(retrieved_sources) for s in case.expect_sources_contains) if case.expect_sources_contains else True

        case_passed = all(checks.values())
        if case_passed:
            passed += 1

        audit.log("EVAL_CASE_RESULT", case_trace_id, {
            "batch_id": batch_id, "case_id": case_id,
            "passed": case_passed, "blocked": blocked,
            "confidence": confidence, "checks": checks,
            "sources": retrieved_sources, "error": error_msg
        })

        result = {
            "case_id": case_id, "trace_id": case_trace_id,
            "role": case.role, "question": case.question,
            "passed": case_passed, "blocked": blocked,
            "confidence": confidence, "checks": checks, "error": error_msg
        }
        if body.include_sources: result["sources"] = retrieved_sources
        if body.include_answer:  result["answer"]  = answer
        results.append(result)

    total     = len(body.cases)
    pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0
    summary   = {
        "batch_id": batch_id, "total": total,
        "passed": passed, "failed": total - passed,
        "pass_rate_pct": pass_rate, "results": results
    }

    audit.log("EVAL_BATCH_DONE", batch_id, {
        "total": total, "passed": passed,
        "failed": total - passed, "pass_rate_pct": pass_rate
    })
    return summary


# ============================================================
# 7️⃣ CACHE ENDPOINTS
# ============================================================

@app.get("/secure-rag/cache/stats", tags=["Internal"])
async def cache_stats(x_eval_key: Optional[str] = Header(default=None, alias="x-eval-key")):
    _require_eval_key(x_eval_key)
    stats = llm_cache.stats()
    audit.log("CACHE_STATS_REQUESTED", "system", stats)
    return stats


@app.delete("/secure-rag/cache/clear", tags=["Internal"])
async def cache_clear(x_eval_key: Optional[str] = Header(default=None, alias="x-eval-key")):
    _require_eval_key(x_eval_key)
    cleared = llm_cache.clear()
    audit.log("CACHE_CLEARED", "system", {"entries_removed": cleared})
    return {"status": "cleared", "entries_removed": cleared}


# ============================================================
# 8️⃣ HEALTH CHECKS
# ============================================================

@app.get("/", tags=["Health"])
def health():
    return {
        "status":      "ok",
        "service":     "Tech Secure RAG",
        "version":     "3.0.0",
        "auth_loaded": AUTH_AVAILABLE,
        "endpoint":    "/secure-rag/invoke"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    health_status = {
        "status": "ok", "service": "Tech Secure RAG",
        "version": "3.0.0", "components": {}
    }

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

    try:
        _embeddings.embed_query("health check")
        health_status["components"]["embeddings"] = "healthy"
    except Exception as e:
        health_status["components"]["embeddings"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    try:
        _llm.invoke("ping")
        health_status["components"]["llm"] = "healthy"
    except Exception as e:
        health_status["components"]["llm"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    if all(v == "healthy" for v in health_status["components"].values()):
        health_status["status"] = "healthy"

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)