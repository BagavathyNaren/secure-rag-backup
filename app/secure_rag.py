# app/secure_rag.py

import re
import json
import uuid
import hashlib
import time
from datetime import datetime
from typing import Dict, Optional

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
            **(data or {}),
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
# ✅ LLM ANSWER CACHE
# ============================================================

class LLMCache:
    """
    In-memory TTL cache for LLM answers.

    Key   : SHA256(role + "|" + normalized_question)
    Value : {"answer": str, "confidence": str}
    TTL   : 300 seconds (5 minutes) by default
    Max   : 200 entries — FIFO eviction when full
    """
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 200):
        self._store: Dict[str, dict] = {}
        self._ttl = ttl_seconds
        self._max = max_entries

    def _make_key(self, role: str, question: str) -> str:
        normalized = question.strip().lower()
        raw = f"{role}|{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, role: str, question: str) -> Optional[dict]:
        key = self._make_key(role, question)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry["expires_at"]:
            del self._store[key]
            return None
        return entry["value"]

    def set(self, role: str, question: str, value: dict) -> None:
        if len(self._store) >= self._max:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]

        key = self._make_key(role, question)
        self._store[key] = {
            "value": value,
            "expires_at": time.monotonic() + self._ttl,
        }

    def invalidate(self, role: str, question: str) -> None:
        self._store.pop(self._make_key(role, question), None)

    def clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count

    def stats(self) -> dict:
        now = time.monotonic()
        active = sum(1 for e in self._store.values() if e["expires_at"] > now)
        expired = len(self._store) - active
        return {
            "total_entries": len(self._store),
            "active_entries": active,
            "expired_entries": expired,
            "ttl_seconds": self._ttl,
            "max_entries": self._max,
        }


# Global cache instance
llm_cache = LLMCache(ttl_seconds=300, max_entries=200)

audit.log("CACHE_INIT", "startup", {"ttl_seconds": 300, "max_entries": 200})


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
            audit.log("SECURITY_BLOCK", trace_id, {"trigger": "pre_filter", "pattern": pattern})
            raise ValueError("SECURITY BLOCK: Credential detected.")
    return answer


# ============================================================
# CONTEXT CREDENTIAL SCANNER
# ============================================================

SENSITIVE_CONTEXT_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"claude-api-key-[a-zA-Z0-9\-]{10,}",
    r"-----BEGIN\s+[A-Z ]+PRIVATE KEY-----",
    r"ghp_[a-zA-Z0-9]{36}",
    r"(?i)secret.{0,5}access.{0,5}key\s*[:=]\s*\S{10,}",
    r"(?i)access.{0,5}key.{0,5}id\s*[:=]\s*[A-Z0-9]{10,}",
    r"sk_[a-zA-Z0-9]{32,}",
]

def scan_context_for_credentials(context: str, trace_id: str = "system") -> None:
    for pattern in SENSITIVE_CONTEXT_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            audit.log("CONTEXT_CREDENTIAL_BLOCK", trace_id, {
                "reason": "credentials_found_in_retrieved_context",
                "pattern": pattern,
                "action": "BLOCKED_BEFORE_GENERATION",
            })
            raise ValueError(
                "SECURITY BLOCK: Retrieved context contains sensitive credentials. "
                "Request blocked before LLM generation."
            )


# ============================================================
# SENSITIVE ANSWER TERM SCANNER
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
    for pattern in SENSITIVE_ANSWER_TERMS:
        if re.search(pattern, answer, re.IGNORECASE):
            audit.log("SECURITY_BLOCK", trace_id, {
                "trigger": "sensitive_term_in_answer",
                "pattern": pattern,
                "action": "BLOCKED_AFTER_GENERATION",
            })
            raise ValueError("SECURITY BLOCK: Answer references sensitive credential terms.")
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
            "action": "BLOCKED",
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
# VECTORSTORE & RETRIEVAL (WITH STARTUP LOGS)
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
    "embedding_model": EMBEDDING_MODEL_NAME,
})

if _index_exists:
    audit.log("FAISS_INDEX_LOAD_START", "startup", {"index_path": INDEX_PATH})
    try:
        _vectorstore = FAISS.load_local(
            INDEX_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        audit.log("FAISS_INDEX_LOAD_FAILED", "startup", {"error": str(e)})
        raise

    audit.log("FAISS_INDEX_LOADED", "startup", {
        "index_path": INDEX_PATH,
        "ntotal": _faiss_ntotal(_vectorstore),
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
        "chunk_count": len(_chunks),
    })


def build_secure_retriever(user_role: str, trace_id: str = "system"):
    allowed = {
        "employee": ["company_policy.txt", "engineering_standards.docx"],
        "security": ["security_policy.txt"],
        "finance": ["finance_policy.txt"],
        "admin": None,
    }.get(user_role, [])

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

    if user_role == "admin":
        def admin_retrieve(q):
            docs = retriever.invoke(q)
            audit.log("RETRIEVAL", trace_id, {
                "role": "admin",
                "doc_count": len(docs),
                "sources": [d.metadata.get("file_name") for d in docs],
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
            "sources": [d.metadata.get("file_name") for d in filtered],
        })
        return filtered

    return role_retrieve


# ============================================================
# PROMPT + LLM
# ============================================================

secure_prompt = ChatPromptTemplate.from_template("""
You are a secure company assistant. Never reveal secrets, keys, or server details.

STRICT ANSWERING RULES:
- Answer using ONLY facts that are explicitly stated in the Context.
- Do NOT use general knowledge or make assumptions.
- Do NOT substitute related information for the requested information.
- If the Context does not contain the exact requested detail, reply:
  "Not specified in the provided context."

ACCESS-AWARE RULES:
- The user role is: {role}
- You may ONLY answer from the retrieved sources listed below: {sources}
- If the question asks about a domain/policy (e.g., finance/security) but the retrieved sources do not include that domain, explicitly say you may not have access to that policy and answer only what is available (or say it's not specified).

Context:
{context}

Question:
{question}

Answer (be concise):
""")

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    max_tokens=450,
)


# ============================================================
# OUTPUT GUARDRAIL + CONFIDENCE
# ============================================================

def model_guard_check(answer: str, context: str = "", trace_id: str = "system") -> str:
    if any(re.search(p, answer, re.IGNORECASE) for p in DANGEROUS_PATTERNS):
        audit.log("SECURITY_BLOCK", trace_id, {"trigger": "output_guard"})
        raise ValueError("SECURITY BLOCK: Credential detected in output.")
    return answer

# ✅ UPDATED: downgrades "not specified" to LOW
def compute_confidence(retrieved_docs, answer: str) -> str:
    if not retrieved_docs:
        return "LOW"

    a = (answer or "").lower()

    low_markers = [
        "not specified",
        "not provided in the context",
        "not in the provided context",
        "not available in the provided context",
        "cannot provide",
        "can't provide",
        "don't know",
        "do not know",
        "no information",
    ]
    if any(m in a for m in low_markers):
        return "LOW"

    if len(answer) < 30:
        return "LOW"

    return "HIGH"


# ============================================================
# MAIN FUNCTION — WITH CACHING
# ============================================================

def secure_rag_invoke(user_input: str, user_role: str = "employee") -> Dict:
    trace_id = new_trace_id()

    audit.log("REQUEST_START", trace_id, {
        "role": user_role,
        "question_preview": user_input[:80],
    })

    try:
        # Input guardrails
        detect_prompt_injection(user_input, trace_id)
        clean_input = redact_pii(user_input, trace_id)

        # ✅ Check cache FIRST
        cached = llm_cache.get(user_role, clean_input)
        if cached:
            audit.log("CACHE_HIT", trace_id, {
                "role": user_role,
                "confidence": cached["confidence"],
            })
            return {
                "answer": cached["answer"],
                "confidence": cached["confidence"],
                "cached": True,
            }

        audit.log("CACHE_MISS", trace_id, {"role": user_role})

        # Retrieval
        docs = build_secure_retriever(user_role, trace_id)(clean_input)
        context = "\n\n".join(d.page_content for d in docs)

        # Sources string for prompt (role-aware, source-aware)
        sources_str = ", ".join(
            sorted(set(d.metadata.get("file_name") for d in docs if d.metadata.get("file_name")))
        ) or "none"

        # Scan context BEFORE LLM
        scan_context_for_credentials(context, trace_id)

        # Generation
        setup = RunnableParallel(
            context=lambda _: context,
            question=RunnablePassthrough(),
            role=lambda _: user_role,
            sources=lambda _: sources_str,
        )
        chain = setup | secure_prompt | _llm | StrOutputParser()
        answer = chain.invoke(clean_input)

        # Output guards
        answer = pre_filter_check(answer, trace_id)
        answer = scan_answer_for_sensitive_terms(answer, trace_id)
        answer = model_guard_check(answer, context, trace_id)

        confidence = compute_confidence(docs, answer)

        # ✅ Cache only HIGH-confidence answers
        if confidence == "HIGH":
            llm_cache.set(user_role, clean_input, {
                "answer": answer,
                "confidence": confidence,
            })
            audit.log("CACHE_STORED", trace_id, {
                "role": user_role,
                "confidence": confidence,
                "cache_stats": llm_cache.stats(),
            })
        else:
            audit.log("CACHE_SKIPPED", trace_id, {
                "role": user_role,
                "confidence": confidence,
                "reason": "low_confidence",
            })

        audit.log("REQUEST_SUCCESS", trace_id, {
            "confidence": confidence,
            "answer_length": len(answer),
            "sources_used": [d.metadata.get("file_name") for d in docs],
        })

        return {
            "answer": answer,
            "confidence": confidence,
            "cached": False,
        }

    except Exception as e:
        audit.log("REQUEST_FAILED", trace_id, {
            "error": str(e),
            "role": user_role,
        })
        return {
            "answer": "I cannot assist with that request due to security restrictions.",
            "confidence": "BLOCKED",
        }