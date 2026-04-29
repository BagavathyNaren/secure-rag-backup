# app/secure_rag.py

import re
import json
import uuid
from datetime import datetime
from typing import Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from app.config import *
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking


# ============================================================
# AUDIT LOGGER — PER-REQUEST TRACE ID
# ============================================================

class AuditLogger:
    """
    Each call to secure_rag_invoke() creates a NEW trace_id.
    This gives us full per-request tracing in the logs.
    """

    def log(self, event: str, trace_id: str, data: dict = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,       # Unique per request
            "event": event,
            **(data or {})
        }
        print(json.dumps(entry), flush=True)

audit = AuditLogger()

def new_trace_id() -> str:
    """Generate a fresh unique trace ID for every request"""
    return str(uuid.uuid4())


# ============================================================
# BACKWARDS COMPATIBLE log_event
# server.py calls: log_event("ANSWER", some_string)
# ============================================================

def log_event(event_type: str, data):
    """Legacy compatibility wrapper for server.py"""
    if isinstance(data, str):
        audit.log(event_type, trace_id="legacy", data={"message": data})
    elif isinstance(data, dict):
        audit.log(event_type, trace_id="legacy", data=data)
    else:
        audit.log(event_type, trace_id="legacy", data={"data": str(data)})


# ============================================================
# NUCLEAR BLOCKS
# ============================================================

DANGEROUS_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"sk_[a-zA-Z0-9]{32,}",
    r"claude-api-key",
    r"-----BEGIN",
    r"Secret Access Key",
    r"ghp_",
]

def pre_filter_check(answer: str, trace_id: str) -> str:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            audit.log("SECURITY_BLOCK", trace_id, {
                "trigger": "pre_filter",
                "pattern": pattern
            })
            raise ValueError("SECURITY BLOCK: Credential detected.")
    return answer


# ============================================================
# INPUT GUARDRAILS
# ============================================================

INJECTION_PATTERNS = [
    r"ignore previous",
    r"disregard",
    r"system prompt",
    r"reveal confidential",
]

def detect_prompt_injection(user_input: str, trace_id: str = "unknown"):
    if any(re.search(p, user_input, re.IGNORECASE) for p in INJECTION_PATTERNS):
        audit.log("PROMPT_INJECTION_DETECTED", trace_id, {
            "input_preview": user_input[:100],
            "action": "BLOCKED"
        })
        raise ValueError("Prompt injection detected.")

def redact_pii(text: str, trace_id: str = "unknown") -> str:
    original = text
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD]", text)
    if text != original:
        audit.log("PII_REDACTED", trace_id, {"note": "PII found and redacted"})
    return text


# ============================================================
# VECTORSTORE & RETRIEVAL
# ============================================================

INDEX_PATH = "faiss_index"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(INDEX_PATH):
    _vectorstore = FAISS.load_local(
        INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )
else:
    docs = ingest_all()
    chunks = recursive_character_chunking(docs, chunk_size=600, chunk_overlap=150)
    _vectorstore = FAISS.from_documents(chunks, _embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    _vectorstore.save_local(INDEX_PATH)

def build_secure_retriever(user_role: str, trace_id: str):
    allowed = {
        "employee": ["company_policy.txt", "engineering_standards.docx"],
        "security": ["security_policy.txt"],
        "finance":  ["finance_policy.txt"],
        "admin":    None,
    }.get(user_role, [])

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

    if user_role == "admin":
        def admin_retrieve(q):
            docs = retriever.invoke(q)
            audit.log("RETRIEVAL", trace_id, {
                "role": "admin",
                "doc_count": len(docs),
                "sources": [d.metadata.get("file_name") for d in docs]
            })
            return docs
        return admin_retrieve

    def role_retrieve(q):
        docs = retriever.invoke(q)
        filtered = [d for d in docs if d.metadata.get("file_name") in allowed]
        audit.log("RETRIEVAL", trace_id, {
            "role": user_role,
            "allowed_sources": allowed,
            "retrieved": len(docs),
            "returned": len(filtered),
            "sources": [d.metadata.get("file_name") for d in filtered]
        })
        return filtered
    return role_retrieve


# ============================================================
# PROMPT + LLM
# ============================================================

secure_prompt = ChatPromptTemplate.from_template("""
You are a secure company assistant. Never reveal secrets, keys, or server details.

Context:
{context}

Question: {question}

Answer (be concise):
""")

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    max_tokens=450
)


# ============================================================
# OUTPUT GUARDRAIL + CONFIDENCE
# ============================================================

def model_guard_check(answer: str, trace_id: str, context: str = "") -> str:
    if any(re.search(p, answer, re.IGNORECASE) for p in DANGEROUS_PATTERNS):
        audit.log("SECURITY_BLOCK", trace_id, {"trigger": "output_guard"})
        raise ValueError("SECURITY BLOCK: Credential detected in output.")
    return answer

def compute_confidence(retrieved_docs, answer: str) -> str:
    if not retrieved_docs:
        return "LOW"
    if len(answer) < 30:
        return "LOW"
    if any(p in answer.lower() for p in ["cannot", "don't know", "no information"]):
        return "LOW"
    return "HIGH"


# ============================================================
# MAIN FUNCTION — FULL PER-REQUEST TRACE ID
# ============================================================

def secure_rag_invoke(user_input: str, user_role: str = "employee") -> Dict:

    # ✅ Fresh unique trace ID for EVERY request
    trace_id = new_trace_id()

    audit.log("REQUEST_START", trace_id, {
        "role": user_role,
        "question_preview": user_input[:80]
    })

    try:
        # Input Guardrails
        detect_prompt_injection(user_input, trace_id)
        clean_input = redact_pii(user_input, trace_id)

        # Retrieval
        docs = build_secure_retriever(user_role, trace_id)(clean_input)
        context = "\n\n".join(d.page_content for d in docs)

        # Generation
        chain = (
            RunnableParallel(
                context=lambda _: context,
                question=RunnablePassthrough()
            )
            | secure_prompt
            | _llm
            | StrOutputParser()
        )
        answer = chain.invoke(clean_input)

        # Triple Nuclear Defense
        answer = pre_filter_check(answer, trace_id)
        answer = model_guard_check(answer, trace_id, context)

        confidence = compute_confidence(docs, answer)

        audit.log("REQUEST_SUCCESS", trace_id, {
            "confidence": confidence,
            "answer_length": len(answer),
            "sources_used": [d.metadata.get("file_name") for d in docs]
        })

        return {"answer": answer, "confidence": confidence}

    except Exception as e:
        audit.log("REQUEST_FAILED", trace_id, {
            "error": str(e),
            "role": user_role
        })
        return {
            "answer": "I cannot assist with that request due to security restrictions.",
            "confidence": "BLOCKED"
        }