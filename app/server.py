# app/server.py

from fastapi import FastAPI, HTTPException, Request, Header
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

from app.secure_rag import (
    detect_prompt_injection,
    redact_pii,
    build_secure_retriever,
    secure_prompt,
    _llm,
    model_guard_check,
    compute_confidence,
    log_event,
    audit,
    new_trace_id,
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
# 3️⃣ REQUEST MODELS
# ============================================================

class SecureRAGRequest(BaseModel):
    question: str
    role: str = "employee"


class SecureRAGResponse(BaseModel):
    answer: str
    confidence: str


# ============================================================
# 3b️⃣ EVALUATION MODELS
# ============================================================

class EvalCase(BaseModel):
    id: Optional[str] = None
    question: str
    role: str = "employee"
    should_block: bool = False
    expect_confidence: Optional[str] = Field(
        default=None,
        description="Expected confidence level: HIGH, LOW, or BLOCKED"
    )
    expect_answer_contains: List[str] = Field(
        default_factory=list,
        description="List of substrings expected in the answer"
    )
    expect_sources_contains: List[str] = Field(
        default_factory=list,
        description="List of source filenames expected to be retrieved"
    )


class EvalRequest(BaseModel):
    cases: List[EvalCase]
    include_answer: bool = Field(
        default=False,
        description="Include full answer text in response"
    )
    include_sources: bool = Field(
        default=True,
        description="Include retrieved source filenames in response"
    )


# ============================================================
# EVAL AUTH GUARD
# ============================================================

EVAL_API_KEY = os.getenv("EVAL_API_KEY", "")

def _require_eval_key(x_eval_key: Optional[str]):
    """
    Protects the eval endpoint.
    Set EVAL_API_KEY in HF Spaces secrets.
    Pass it as header: x-eval-key: <your-key>
    """
    if not EVAL_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Evaluation endpoint is disabled. Set EVAL_API_KEY secret to enable it."
        )
    if not x_eval_key or x_eval_key != EVAL_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: invalid or missing x-eval-key header."
        )


# ============================================================
# 4️⃣ SECURE RAG ENDPOINT (STREAMING)
# ============================================================

@app.post("/secure-rag/invoke")
@limiter.limit("10/minute")
async def secure_rag_endpoint(request: Request, body: SecureRAGRequest):

    trace_id = new_trace_id()

    audit.log("REQUEST_START", trace_id, {
        "role": body.role,
        "question_preview": body.question[:120]
    })

    try:
        # Input Guardrails
        detect_prompt_injection(body.question, trace_id)
        user_input = redact_pii(body.question, trace_id)

        # Secure Retrieval
        retriever = build_secure_retriever(body.role, trace_id)
        retrieved_docs = retriever(user_input)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Streaming Chain
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

            # Output Guardrails
            try:
                full_answer = model_guard_check(full_answer, context, trace_id)
                confidence = compute_confidence(retrieved_docs, full_answer)

                audit.log("ANSWER", trace_id, {
                    "message": full_answer[:500],
                    "confidence": confidence,
                    "role": body.role,
                    "sources": [d.metadata.get("file_name") for d in retrieved_docs]
                })

                yield f"data: {json.dumps({'done': True, 'answer': full_answer, 'confidence': confidence})}\n\n"

            except ValueError as ve:
                audit.log("REQUEST_FAILED", trace_id, {
                    "error": str(ve),
                    "stage": "output_guard"
                })
                yield f"data: {json.dumps({'error': str(ve)})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    except ValueError as ve:
        audit.log("REQUEST_FAILED", trace_id, {
            "error": str(ve),
            "stage": "input_guard"
        })
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        audit.log("REQUEST_FAILED", trace_id, {
            "error": str(e),
            "stage": "unhandled"
        })
        logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ============================================================
# 4b️⃣ EVALUATION ENDPOINT (INTERNAL TESTING)
# ============================================================

@app.post("/secure-rag/eval")
@limiter.limit("5/minute")
async def secure_rag_eval(
    request: Request,
    body: EvalRequest,
    x_eval_key: Optional[str] = Header(default=None, alias="x-eval-key"),
):
    # Auth check
    _require_eval_key(x_eval_key)

    batch_id = new_trace_id()

    audit.log("EVAL_BATCH_START", batch_id, {
        "total_cases": len(body.cases)
    })

    results = []
    passed = 0

    for i, case in enumerate(body.cases):
        case_trace_id = new_trace_id()
        case_id = case.id or f"case_{i + 1}"

        audit.log("EVAL_CASE_START", case_trace_id, {
            "batch_id": batch_id,
            "case_id": case_id,
            "role": case.role,
            "question_preview": case.question[:120],
            "should_block": case.should_block
        })

        # State for this case
        answer = None
        confidence = None
        retrieved_sources = []
        blocked = False
        error_msg = None

        try:
            # Step 1: Input guardrails
            detect_prompt_injection(case.question, case_trace_id)
            clean_input = redact_pii(case.question, case_trace_id)

            # Step 2: Retrieval
            retriever = build_secure_retriever(case.role, case_trace_id)
            retrieved_docs = retriever(clean_input)
            retrieved_sources = [d.metadata.get("file_name") for d in retrieved_docs]
            context = "\n\n".join(d.page_content for d in retrieved_docs)

            # Step 3: Generation (non-streaming for eval)
            setup = RunnableParallel(
                context=lambda _: context,
                question=RunnablePassthrough()
            )
            rag_chain = setup | secure_prompt | _llm | StrOutputParser()
            answer = await rag_chain.ainvoke(clean_input)

            # Step 4: Output guard
            answer = model_guard_check(answer, context, case_trace_id)
            confidence = compute_confidence(retrieved_docs, answer)

        except Exception as e:
            blocked = True
            error_msg = str(e)
            confidence = "BLOCKED"

        # ============================================================
        # ASSERTIONS — check against expectations
        # ============================================================
        checks = {}

        # 1. Was it blocked as expected?
        checks["block_check"] = (blocked == case.should_block)

        # 2. Confidence matches expectation?
        if case.expect_confidence:
            checks["confidence_check"] = (confidence == case.expect_confidence)
        else:
            checks["confidence_check"] = True     # no expectation = auto pass

        # 3. Answer contains expected substrings?
        if case.expect_answer_contains and answer:
            answer_lower = answer.lower()
            checks["answer_content_check"] = all(
                keyword.lower() in answer_lower
                for keyword in case.expect_answer_contains
            )
        elif case.expect_answer_contains and not answer:
            checks["answer_content_check"] = False  # expected answer but got none
        else:
            checks["answer_content_check"] = True   # no expectation = auto pass

        # 4. Expected sources were retrieved?
        if case.expect_sources_contains:
            source_set = set(retrieved_sources)
            checks["sources_check"] = all(
                s in source_set
                for s in case.expect_sources_contains
            )
        else:
            checks["sources_check"] = True          # no expectation = auto pass

        # Overall pass/fail for this case
        case_passed = all(checks.values())
        if case_passed:
            passed += 1

        audit.log("EVAL_CASE_RESULT", case_trace_id, {
            "batch_id": batch_id,
            "case_id": case_id,
            "passed": case_passed,
            "blocked": blocked,
            "confidence": confidence,
            "checks": checks,
            "sources": retrieved_sources,
            "error": error_msg
        })

        # Build result payload
        result = {
            "case_id": case_id,
            "trace_id": case_trace_id,
            "role": case.role,
            "question": case.question,
            "passed": case_passed,
            "blocked": blocked,
            "confidence": confidence,
            "checks": checks,
            "error": error_msg
        }

        if body.include_sources:
            result["sources"] = retrieved_sources

        if body.include_answer:
            result["answer"] = answer

        results.append(result)

    # Final summary
    total = len(body.cases)
    failed = total - passed
    pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0

    summary = {
        "batch_id": batch_id,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate_pct": pass_rate,
        "results": results
    }

    audit.log("EVAL_BATCH_DONE", batch_id, {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate_pct": pass_rate
    })

    return summary


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

    if all(v == "healthy" for v in health_status["components"].values()):
        health_status["status"] = "healthy"

    return health_status


# ============================================================
# RUN (local dev)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)