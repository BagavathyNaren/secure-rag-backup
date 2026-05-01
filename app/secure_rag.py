# app/secure_rag.py

import re
import json
import uuid
import hashlib
import time
import os
from datetime import datetime
from typing import Dict, Optional, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import *  # keep as-is if you rely on it elsewhere
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking


# ============================================================
# AUDIT LOGGER — PER-REQUEST TRACE ID
# ============================================================

class AuditLogger:
    def log(self, event: str, trace_id: str = "system", data: dict | None = None):
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
# SETTINGS (read env, but do NOT initialize here)
# ============================================================

REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "secure_rag:cache:v1:")

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")


# ============================================================
# GLOBALS (initialized during FastAPI startup)
# ============================================================

llm_cache = None            # type: ignore[assignment]
_embeddings = None           # type: ignore[assignment]
_vectorstore = None          # type: ignore[assignment]
_llm = None                 # type: ignore[assignment]


# ============================================================
# CACHE BACKENDS
# ============================================================

class InMemoryLLMCache:
    """
    In-memory TTL cache (fallback). Same interface as RedisLLMCache.
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
        self._store[key] = {"value": value, "expires_at": time.monotonic() + self._ttl}

    def invalidate(self, role: str, question: str) -> None:
        self._store.pop(self._make_key(role, question), None)

    def clear(self) -> int:
        n = len(self._store)
        self._store.clear()
        return n

    def stats(self) -> dict:
        now = time.monotonic()
        active = sum(1 for e in self._store.values() if e["expires_at"] > now)
        expired = len(self._store) - active
        return {
            "backend": "memory",
            "total_entries": len(self._store),
            "active_entries": active,
            "expired_entries": expired,
            "ttl_seconds": self._ttl,
            "max_entries": self._max,
        }


class RedisLLMCache:
    """
    Redis TTL cache.

    Keys:  <REDIS_PREFIX><sha256(role|question)>
    Value: JSON string {"answer": "...", "confidence": "HIGH"}
    TTL:   redis expire (seconds)
    """
    def __init__(self, redis_url: str, ttl_seconds: int, prefix: str):
        import redis  # requires redis==5.x

        self._ttl = ttl_seconds
        self._prefix = prefix

        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
            health_check_interval=30,
        )

        # Fail fast; caller can fallback
        self._client.ping()

    def _make_key(self, role: str, question: str) -> str:
        normalized = question.strip().lower()
        raw = f"{role}|{normalized}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return f"{self._prefix}{digest}"

    def get(self, role: str, question: str) -> Optional[dict]:
        key = self._make_key(role, question)
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            self._client.delete(key)
            return None

    def set(self, role: str, question: str, value: dict) -> None:
        key = self._make_key(role, question)
        self._client.setex(key, self._ttl, json.dumps(value))

    def invalidate(self, role: str, question: str) -> None:
        key = self._make_key(role, question)
        self._client.delete(key)

    def clear(self) -> int:
        pattern = f"{self._prefix}*"
        deleted = 0
        pipe = self._client.pipeline(transaction=False)

        batch = 0
        for k in self._client.scan_iter(match=pattern, count=500):
            pipe.delete(k)
            batch += 1
            if batch >= 500:
                deleted += sum(pipe.execute())
                batch = 0

        if batch:
            deleted += sum(pipe.execute())

        return int(deleted)

    def stats(self) -> dict:
        pattern = f"{self._prefix}*"
        count = 0
        for _ in self._client.scan_iter(match=pattern, count=500):
            count += 1
            if count >= 5000:
                break
        return {
            "backend": "redis",
            "prefix": self._prefix,
            "ttl_seconds": self._ttl,
            "approx_entries": count,
        }


# ============================================================
# INIT FUNCTIONS (production-grade: called in FastAPI startup)
# ============================================================

def init_cache(force: bool = False) -> Any:
    """
    Initialize llm_cache.
    - Redis if configured and reachable
    - else fallback to in-memory
    """
    global llm_cache

    if llm_cache is not None and not force:
        audit.log("CACHE_INIT_SKIPPED", "startup", {"reason": "already_initialized"})
        return llm_cache

    try:
        if REDIS_URL:
            llm_cache = RedisLLMCache(REDIS_URL, REDIS_TTL_SECONDS, REDIS_PREFIX)
            audit.log("CACHE_INIT", "startup", {
                "backend": "redis",
                "ttl_seconds": REDIS_TTL_SECONDS,
                "prefix": REDIS_PREFIX,
            })
        else:
            llm_cache = InMemoryLLMCache(ttl_seconds=REDIS_TTL_SECONDS, max_entries=200)
            audit.log("CACHE_INIT", "startup", {
                "backend": "memory",
                "ttl_seconds": REDIS_TTL_SECONDS,
                "max_entries": 200,
            })
    except Exception as e:
        llm_cache = InMemoryLLMCache(ttl_seconds=REDIS_TTL_SECONDS, max_entries=200)
        audit.log("CACHE_INIT_FALLBACK", "startup", {
            "backend": "memory",
            "reason": "redis_unavailable",
            "error": str(e),
            "ttl_seconds": REDIS_TTL_SECONDS,
            "max_entries": 200,
        })

    return llm_cache


def init_embeddings(force: bool = False) -> OpenAIEmbeddings:
    global _embeddings

    if _embeddings is not None and not force:
        audit.log("EMBEDDINGS_INIT_SKIPPED", "startup", {"reason": "already_initialized"})
        return _embeddings

    _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    audit.log("EMBEDDINGS_INIT", "startup", {"embedding_model": EMBEDDING_MODEL_NAME})
    return _embeddings


def _faiss_ntotal(vs) -> int:
    try:
        return int(vs.index.ntotal)
    except Exception:
        return -1


def init_vectorstore(force_rebuild: bool = False) -> FAISS:
    """
    Initialize FAISS vectorstore:
    - If index exists and not force_rebuild -> load
    - Else -> build from documents, then save
    """
    global _vectorstore

    init_embeddings()

    index_exists = os.path.exists(INDEX_PATH)

    audit.log("FAISS_INDEX_CHECK", "startup", {
        "index_path": INDEX_PATH,
        "exists": index_exists,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "force_rebuild": force_rebuild,
    })

    if index_exists and not force_rebuild:
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
        return _vectorstore

    audit.log("FAISS_INDEX_BUILD_START", "startup", {"index_path": INDEX_PATH})

    documents = ingest_all()
    chunks = recursive_character_chunking(documents, chunk_size=600, chunk_overlap=150)
    _vectorstore = FAISS.from_documents(chunks, _embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    _vectorstore.save_local(INDEX_PATH)

    audit.log("FAISS_INDEX_BUILT_AND_SAVED", "startup", {
        "index_path": INDEX_PATH,
        "ntotal": _faiss_ntotal(_vectorstore),
        "chunk_count": len(chunks),
    })
    return _vectorstore


def init_llm(force: bool = False) -> ChatOpenAI:
    global _llm

    if _llm is not None and not force:
        audit.log("LLM_INIT_SKIPPED", "startup", {"reason": "already_initialized"})
        return _llm

    _llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0,
        streaming=True,
        max_tokens=450,
    )
    audit.log("LLM_INIT", "startup", {"model": LLM_MODEL_NAME})
    return _llm


def ensure_initialized() -> None:
    missing = []
    if llm_cache is None:
        missing.append("llm_cache")
    if _embeddings is None:
        missing.append("_embeddings")
    if _vectorstore is None:
        missing.append("_vectorstore")
    if _llm is None:
        missing.append("_llm")

    if missing:
        raise RuntimeError(
            "Secure RAG is not initialized. Missing: "
            + ", ".join(missing)
            + ". Ensure startup calls init_cache/init_vectorstore/init_llm."
        )


# ============================================================
# GUARDRAILS (unchanged)
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
# VECTORSTORE & RETRIEVAL (no startup side effects)
# ============================================================

def build_secure_retriever(user_role: str, trace_id: str = "system"):
    if _vectorstore is None:
        raise RuntimeError("Vectorstore not initialized. Call init_vectorstore() during startup.")

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
# PROMPT (safe at import time)
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


# ============================================================
# OUTPUT GUARDRAIL + CONFIDENCE (unchanged)
# ============================================================

def model_guard_check(answer: str, context: str = "", trace_id: str = "system") -> str:
    if any(re.search(p, answer, re.IGNORECASE) for p in DANGEROUS_PATTERNS):
        audit.log("SECURITY_BLOCK", trace_id, {"trigger": "output_guard"})
        raise ValueError("SECURITY BLOCK: Credential detected in output.")
    return answer


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
# Add shutdown helper
# ============================================================

def shutdown_rag() -> None:
    """
    Best-effort cleanup for external clients (Redis).
    Safe to call multiple times.
    """
    global llm_cache

    try:
        # If Redis backend, close connections cleanly
        if llm_cache is not None and hasattr(llm_cache, "_client"):
            client = getattr(llm_cache, "_client", None)
            if client is not None:
                # redis-py 5 supports close(); keep try/except for compatibility
                try:
                    client.close()
                except Exception:
                    pass

                # extra safe: disconnect pool
                try:
                    client.connection_pool.disconnect()
                except Exception:
                    pass
    except Exception:
        # Never raise during shutdown
        pass

LIFECYCLE_LAST_SHUTDOWN_KEY = "secure_rag:lifecycle:last_shutdown"

def write_last_shutdown_marker(ts: str) -> None:
    """
    Store last shutdown timestamp in Redis (if using Redis cache).
    No-op if cache backend isn't Redis.
    """
    global llm_cache
    try:
        if llm_cache is not None and hasattr(llm_cache, "_client"):
            client = getattr(llm_cache, "_client", None)
            if client is not None:
                client.set(LIFECYCLE_LAST_SHUTDOWN_KEY, ts)
    except Exception:
        pass

def read_last_shutdown_marker() -> str | None:
    """
    Read last shutdown timestamp from Redis (if using Redis cache).
    """
    global llm_cache
    try:
        if llm_cache is not None and hasattr(llm_cache, "_client"):
            client = getattr(llm_cache, "_client", None)
            if client is not None:
                return client.get(LIFECYCLE_LAST_SHUTDOWN_KEY)
    except Exception:
        return None
    return None


# ============================================================
# MAIN FUNCTION — WITH CACHING (optional helper; unchanged logic)
# ============================================================

def secure_rag_invoke(user_input: str, user_role: str = "employee") -> Dict:
    ensure_initialized()

    trace_id = new_trace_id()

    audit.log("REQUEST_START", trace_id, {
        "role": user_role,
        "question_preview": user_input[:80],
    })

    try:
        detect_prompt_injection(user_input, trace_id)
        clean_input = redact_pii(user_input, trace_id)

        cached = llm_cache.get(user_role, clean_input)  # type: ignore[union-attr]
        if cached:
            audit.log("CACHE_HIT", trace_id, {
                "role": user_role,
                "confidence": cached.get("confidence"),
            })
            return {
                "answer": cached["answer"],
                "confidence": cached["confidence"],
                "cached": True,
            }

        audit.log("CACHE_MISS", trace_id, {"role": user_role})

        docs = build_secure_retriever(user_role, trace_id)(clean_input)
        context = "\n\n".join(d.page_content for d in docs)

        sources_str = ", ".join(
            sorted(set(d.metadata.get("file_name") for d in docs if d.metadata.get("file_name")))
        ) or "none"

        scan_context_for_credentials(context, trace_id)

        setup = RunnableParallel(
            context=lambda _: context,
            question=RunnablePassthrough(),
            role=lambda _: user_role,
            sources=lambda _: sources_str,
        )

        chain = setup | secure_prompt | _llm | StrOutputParser()  # type: ignore[arg-type]
        answer = chain.invoke(clean_input)

        answer = pre_filter_check(answer, trace_id)
        answer = scan_answer_for_sensitive_terms(answer, trace_id)
        answer = model_guard_check(answer, context, trace_id)

        confidence = compute_confidence(docs, answer)

        if confidence == "HIGH":
            llm_cache.set(user_role, clean_input, {  # type: ignore[union-attr]
                "answer": answer,
                "confidence": confidence,
            })
            audit.log("CACHE_STORED", trace_id, {
                "role": user_role,
                "confidence": confidence,
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