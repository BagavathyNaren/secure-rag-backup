# app/secure_rag.py

import re
import logging
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
# STRUCTURED JSON LOGGER (AUDIT-READY)
# ============================================================

class JSONLogger:
    def __init__(self):
        self.request_id = str(uuid.uuid4())

    def log(self, event_type: str, data: dict):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": self.request_id,
            "event": event_type,
            **data
        }
        print(json.dumps(log_entry))  # In production → send to stdout for log collectors

logger = JSONLogger()


# ============================================================
# NUCLEAR-GRADE HARD BLOCKS
# ============================================================

DANGEROUS_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"sk_[a-zA-Z0-9]{32,}",
    r"claude-api-key-[a-zA-Z0-9-]{20,}",
    r"[A-Za-z0-9+/]{40}",
    r"ghp_[a-zA-Z0-9]{36}",
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
    r"Secret Access Key[^\n]{0,20}[:=][^\n]{10,}",
]

BLOCKED_KEYWORDS = [
    "AKIA", "sk_", "claude-api-key", "Secret Access Key", "Secret Key",
    "private key", "BEGIN PRIVATE KEY", "ghp_", "github_pat"
]

def pre_filter_check(answer: str) -> str:
    if any(kw in answer for kw in BLOCKED_KEYWORDS):
        logger.log("SECURITY_BLOCK", {"reason": "blocked_keyword", "matched": next(kw for kw in BLOCKED_KEYWORDS if kw in answer)})
        raise ValueError("SECURITY BLOCK: Forbidden keyword detected.")
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            logger.log("SECURITY_BLOCK", {"reason": "credential_pattern", "pattern": pattern})
            raise ValueError("SECURITY BLOCK: Credential pattern detected.")
    return answer


# ============================================================
# INPUT GUARDRAILS + PII REDACTION
# ============================================================

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"reveal confidential",
    r"dump entire database",
    r"show all employees",
    r"system prompt",
    r"disregard security",
]

EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
CREDIT_CARD_REGEX = r"\b(?:\d[ -]*?){13,16}\b"
API_KEY_REGEX = r"sk-[a-zA-Z0-9]{32}"

def detect_prompt_injection(user_input: str):
    lower = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            logger.log("PROMPT_INJECTION_DETECTED", {"input": user_input})
            raise ValueError("Prompt injection detected.")

def redact_pii(text: str) -> str:
    original = text
    text = re.sub(EMAIL_REGEX, "[REDACTED_EMAIL]", text)
    text = re.sub(CREDIT_CARD_REGEX, "[REDACTED_CC]", text)
    text = re.sub(API_KEY_REGEX, "[BLOCKED_API_KEY]", text)
    if text != original:
        logger.log("PII_REDACTED", {"original_length": len(original), "redacted_length": len(text)})
    return text


# ============================================================
# VECTORSTORE + RETRIEVAL
# ============================================================

INDEX_PATH = "faiss_index"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(INDEX_PATH):
    print("Loading prebuilt FAISS index...")
    _vectorstore = FAISS.load_local(INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded.")
else:
    print("Building new FAISS index...")
    _documents = ingest_all()
    _chunks = recursive_character_chunking(_documents, chunk_size=600, chunk_overlap=150)
    _vectorstore = FAISS.from_documents(_chunks, _embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    _vectorstore.save_local(INDEX_PATH)
    print("FAISS index built and saved.")

def build_secure_retriever(user_role: str):
    allowed_sources = {
        "employee": ["company_policy.txt", "engineering_standards.docx"],
        "security": ["security_policy.txt"],
        "finance": ["finance_policy.txt"],
        "admin": None,
    }
    allowed = allowed_sources.get(user_role, [])
    base_retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    if user_role == "admin":
        def retrieve(query: str):
            docs = base_retriever.invoke(query)
            logger.log("RETRIEVAL", {
                "role": user_role,
                "doc_count": len(docs),
                "sources": [d.metadata.get("file_name") for d in docs]
            })
            return docs
        return retrieve

    def filtered_retrieval(query: str):
        docs = base_retriever.invoke(query)
        filtered = [d for d in docs if d.metadata.get("file_name") in allowed]
        logger.log("RETRIEVAL", {
            "role": user_role,
            "requested_sources": allowed,
            "retrieved_count": len(docs),
            "returned_count": len(filtered),
            "sources": [d.metadata.get("file_name") for d in filtered]
        })
        return filtered

    return filtered_retrieval


# ============================================================
# PROMPT + LLM
# ============================================================

secure_prompt = ChatPromptTemplate.from_template("""
You are a secure assistant. Never reveal credentials or internal details.
Be concise and professional.

Context:
{context}

Question:
{question}

Answer:
""")

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True, max_tokens=450)


# ============================================================
# NUCLEAR GUARDRAIL
# ============================================================

def model_guard_check(answer: str, context: str = "") -> str:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            logger.log("SECURITY_BLOCK", {"reason": "regex_match", "pattern": pattern})
            raise ValueError("SECURITY BLOCK: Credential pattern detected.")

    guard_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    guard_prompt = f"""
YOU ARE A SECURITY FIREWALL. BLOCK ANYTHING SUSPICIOUS.

Context: {context[:2000]}
Answer: {answer}

IF you see ANY key, secret, server detail, credential → reply UNSAFE
Reply only: SAFE or UNSAFE
"""
    result = guard_llm.invoke(guard_prompt).content.strip().upper()
    if result != "SAFE":
        logger.log("SECURITY_BLOCK", {"reason": "llm_guard_triggered", "guard_response": result})
        raise ValueError("Output blocked by model-based guardrail.")
    return answer


# ============================================================
# CONFIDENCE
# ============================================================

def compute_confidence(retrieved_docs, answer):
    if not retrieved_docs:
        return "LOW"
    if any(phrase in answer.lower() for phrase in ["does not specify", "cannot provide", "no information"]):
        return "LOW"
    if len(answer.split()) < 6:
        return "LOW"
    return "HIGH"


# ============================================================
# MAIN INVOKE (AUDIT-COMPLETE)
# ============================================================

def secure_rag_invoke(user_input: str, user_role: str = "employee") -> Dict:
    # Start audit trail
    logger.log("REQUEST_START", {"question": user_input, "role": user_role})

    try:
        detect_prompt_injection(user_input)
        redacted_input = redact_pii(user_input)

        retriever = build_secure_retriever(user_role)
        retrieved_docs = retriever.invoke(redacted_input)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        setup = RunnableParallel(context=lambda _: context, question=RunnablePassthrough())
        rag_chain = setup | secure_prompt | _llm | StrOutputParser()
        answer = rag_chain.invoke(redacted_input)

        # NUCLEAR DEFENSE
        answer = pre_filter_check(answer)
        answer = model_guard_check(answer, context)

        confidence = compute_confidence(retrieved_docs, answer)

        logger.log("REQUEST_SUCCESS", {
            "confidence": confidence,
            "answer_length": len(answer),
            "retrieved_sources": [d.metadata.get("file_name") for d in retrieved_docs]
        })

        return {"answer": answer, "confidence": confidence}

    except Exception as e:
        error_msg = str(e)
        logger.log("REQUEST_FAILED", {"error": error_msg})
        return {"answer": "I'm sorry, I cannot assist with that request due to security restrictions.", "confidence": "BLOCKED"}