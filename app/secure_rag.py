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
    def log(self, event: str, trace_id: str = "system", data: dict = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "event": event,
            **(data or {})
        }
        print(json.dumps(entry), flush=True)

audit = AuditLogger()

def new_trace_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# BACKWARDS COMPATIBLE log_event
# ============================================================

def log_event(event_type: str, data):
    if isinstance(data, str):
        audit.log(event_type, data={"message": data})
    elif isinstance(data, dict):
        audit.log(event_type, data=data)
    else:
        audit.log(event_type, data={"data": str(data)})


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

def pre_filter_check(answer: str, trace_id: str = "system") -> str:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            audit.log("SECURITY_BLOCK", trace_id, {
                "trigger": "pre_filter",
                "pattern": pattern
            })
            raise ValueError("SECURITY BLOCK: Credential detected.")
    return answer


# ============================================================
# ✅ NEW — THING 2
# CONTEXT CREDENTIAL SCANNER
# Scans RAW retrieved context BEFORE sending to LLM.
# If credentials are found → block immediately.
# LLM never sees the sensitive data.
# ============================================================

SENSITIVE_CONTEXT_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",                         # AWS Access Key ID
    r"claude-api-key-[a-zA-Z0-9\-]{10,}",        # Claude API key
    r"-----BEGIN\s+[A-Z ]+PRIVATE KEY-----",      # Private keys
    r"ghp_[a-zA-Z0-9]{36}",                      # GitHub PAT
    r"(?i)secret.{0,5}access.{0,5}key\s*[:=]\s*\S{10,}",  # Secret Access Key: value
    r"(?i)access.{0,5}key.{0,5}id\s*[:=]\s*[A-Z0-9]{10,}",# Access Key ID: value
    r"sk_[a-zA-Z0-9]{32,}",                      # OpenAI/Stripe keys
]

def scan_context_for_credentials(context: str, trace_id: str = "system") -> None:
    """
    Scans the raw retrieved context BEFORE passing to LLM.
    If any credential pattern is found → raise immediately.
    This is the earliest possible block point.
    """
    for pattern in SENSITIVE_CONTEXT_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            audit.log("CONTEXT_CREDENTIAL_BLOCK", trace_id, {
                "reason": "credentials_found_in_retrieved_context",
                "pattern": pattern,
                "action": "BLOCKED_BEFORE_GENERATION"
            })
            raise ValueError(
                "SECURITY BLOCK: Retrieved context contains sensitive credentials. "
                "Request blocked before LLM generation."
            )


# ============================================================
# ✅ NEW — THING 1
# SENSITIVE ANSWER TERM SCANNER
# Blocks answers that DESCRIBE credentials even without raw values.
# e.g. "The document contains a fake AWS access key..." → BLOCKED
# ============================================================

SENSITIVE_ANSWER_TERMS = [
    r"(?i)(api\s*key|secret\s*key|access\s*key|private\s*key)",
    r"(?i)(aws|s3).{0,20}(key|credential|secret|token)",
    r"(?i)(fake|mock|test|simulated|non-functional).{0,40}(key|credential|secret|token)",
    r"(?i)(credential|credentials).{0,30}(fake|mock|test|simulated|demo)",
    r"(?i)claude.{0,10}api.{0,10}key",
    r"(?i)(access\s*key\s*id|secret\s*access\s*key)",
]

def scan_answer_for_sensitive_terms(answer: str, trace_id: str = "system") -> str:
    """
    Scans the LLM answer for descriptions of credentials
    even when raw credential values are not present.
    Catches answers like:
    - 'The document contains a fake AWS access key'
    - 'There is a Claude API key in the file'
    - 'Simulated credentials are present'
    """
    for pattern in SENSITIVE_ANSWER_TERMS:
        if re.search(pattern, answer, re.IGNORECASE):
            audit.log("SECURITY_BLOCK", trace_id, {
                "trigger": "sensitive_term_in_answer",
                "pattern": pattern,
                "action": "BLOCKED_AFTER_GENERATION"
            })
            raise ValueError(
                "SECURITY BLOCK: Answer references sensitive credential terms."
            )
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

def detect_prompt_injection(user_input: str, trace_id: str = "system"):
    if any(re.search(p, user_input, re.IGNORECASE) for p in INJECTION_PATTERNS):
        audit.log("PROMPT_INJECTION_DETECTED", trace_id, {
            "input_preview": user_input[:100],
            "action": "BLOCKED"
        })
        raise ValueError("Prompt injection detected.")

def redact_pii(text: str, trace_id: str = "system") -> str:
    original = text
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD]", text)
    if text != original:
        audit.log("PII_REDACTED", trace_id, {"note": "PII found and redacted"})
    return text


# ============================================================
# VECTORSTORE & RETRIEVAL (+ STARTUP/CACHE LOGS)
# ============================================================

INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

def _faiss_ntotal(vs) -> int:
    try:
        return int(vs.index.ntotal)
    except Exception:
        return -1

_index_exists = os.path.exists(INDEX_PATH)

audit.log("FAISS_INDEX_CHECK", "startup", {
    "index_path": INDEX_PATH,
    "exists": _index_exists,
    "embedding_model": EMBEDDING_MODEL_NAME
})

if _index_exists:
    audit.log("FAISS_INDEX_LOAD_START", "startup", {"index_path": INDEX_PATH})
    try:
        _vectorstore = FAISS.load_local(
            INDEX_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        audit.log("FAISS_INDEX_LOAD_FAILED", "startup", {
            "index_path": INDEX_PATH,
            "error": str(e)
        })
        raise
    audit.log("FAISS_INDEX_LOADED", "startup", {
        "index_path": INDEX_PATH,
        "ntotal": _faiss_ntotal(_vectorstore)
    })
else:
    audit.log("FAISS_INDEX_BUILD_START", "startup", {"index_path": INDEX_PATH})
    _documents = ingest_all()
    _chunks = recursive_character_chunking(_documents, chunk_size=600, chunk_overlap=150)
    _vectorstore = FAISS.from_documents(_chunks, _embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    _vectorstore.save_local(INDEX_PATH)
    audit.log("FAISS_INDEX_BUILT_AND_SAVED", "startup", {
        "index_path": INDEX_PATH,
        "ntotal": _faiss_ntotal(_vectorstore),
        "chunk_count": len(_chunks)
    })


def build_secure_retriever(user_role: str, trace_id: str = "system"):
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

def model_guard_check(answer: str, context: str = "", trace_id: str = "system") -> str:
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
    trace_id = new_trace_id()

    audit.log("REQUEST_START", trace_id, {
        "role": user_role,
        "question_preview": user_input[:80]
    })

    try:
        # Input guardrails
        detect_prompt_injection(user_input, trace_id)
        clean_input = redact_pii(user_input, trace_id)

        # Retrieval
        docs = build_secure_retriever(user_role, trace_id)(clean_input)
        context = "\n\n".join(d.page_content for d in docs)

        # ✅ NEW THING 2 — scan context BEFORE LLM sees it
        scan_context_for_credentials(context, trace_id)

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

        # ✅ NEW THING 1 — block credential descriptions in answer
        answer = pre_filter_check(answer, trace_id)
        answer = scan_answer_for_sensitive_terms(answer, trace_id)  # ← NEW
        answer = model_guard_check(answer, context, trace_id)

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