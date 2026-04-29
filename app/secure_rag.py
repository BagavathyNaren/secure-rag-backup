# app/secure_rag.py

import re
import json
import uuid
from datetime import datetime
from typing import Dict, List

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
# STRUCTURED AUDIT LOGGER (NO log_event ANYMORE)
# ============================================================

class AuditLogger:
    def __init__(self):
        self.request_id = str(uuid.uuid4())

    def log(self, event: str, data: dict = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": self.request_id,
            "event": event,
            **(data or {})
        }
        print(json.dumps(entry), flush=True)

# Global logger — used everywhere
audit = AuditLogger()


# ============================================================
# NUCLEAR BLOCKS
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
        audit.log("SECURITY_BLOCK", {"reason": "keyword", "matched": next(kw for kw in BLOCKED_KEYWORDS if kw in answer)})
        raise ValueError("SECURITY BLOCK: Forbidden keyword detected.")
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            audit.log("SECURITY_BLOCK", {"reason": "regex", "pattern": pattern})
            raise ValueError("SECURITY BLOCK: Credential pattern detected.")
    return answer


# ============================================================
# INPUT GUARDRAILS
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
    if any(re.search(p, user_input, re.IGNORECASE) for p in INJECTION_PATTERNS):
        audit.log("PROMPT_INJECTION", {"input": user_input})
        raise ValueError("Prompt injection detected.")

def redact_pii(text: str) -> str:
    original = text
    text = re.sub(EMAIL_REGEX, "[REDACTED_EMAIL]", text)
    text = re.sub(CREDIT_CARD_REGEX, "[REDACTED_CC]", text)
    text = re.sub(API_KEY_REGEX, "[BLOCKED_API_KEY]", text)
    if text != original:
        audit.log("PII_REDACTED", {"original_length": len(original)})
    return text


# ============================================================
# VECTORSTORE
# ============================================================

INDEX_PATH = "faiss_index"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(INDEX_PATH):
    _vectorstore = FAISS.load_local(INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
else:
    _documents = ingest_all()
    _chunks = recursive_character_chunking(_documents, chunk_size=600, chunk_overlap=150)
    _vectorstore = FAISS.from_documents(_chunks, _embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    _vectorstore.save_local(INDEX_PATH)

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
            audit.log("RETRIEVAL", {"role": "admin", "sources": [d.metadata.get("file_name") for d in docs]})
            return docs
        return retrieve

    def filtered_retrieval(query: str):
        docs = base_retriever.invoke(query)
        filtered = [d for d in docs if d.metadata.get("file_name") in allowed]
        audit.log("RETRIEVAL", {
            "role": user_role,
            "allowed": allowed,
            "returned": [d.metadata.get("file_name") for d in filtered]
        })
        return filtered
    return filtered_retrieval


# ============================================================
# PROMPT + LLM
# ============================================================

secure_prompt = ChatPromptTemplate.from_template("""
You are a secure assistant. Never reveal secrets, keys, or server details.
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

def model_guard_check(answer: str, context: str) -> str:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            audit.log("SECURITY_BLOCK", {"reason": "regex_guard"})
            raise ValueError("SECURITY BLOCK: Credential detected.")
    
    guard = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = guard.invoke(f"Context: {context[:1500]}\nAnswer: {answer}\nIs this safe? Reply only SAFE or UNSAFE").content.strip().upper()
    if result != "SAFE":
        audit.log("SECURITY_BLOCK", {"reason": "llm_guard", "response": result})
        raise ValueError("Output blocked by guardrail.")
    return answer


# ============================================================
# MAIN FUNCTION
# ============================================================

def secure_rag_invoke(user_input: str, user_role: str = "employee") -> Dict:
    audit.log("REQUEST", {"question": user_input, "role": user_role})

    try:
        detect_prompt_injection(user_input)
        clean_input = redact_pii(user_input)

        docs = build_secure_retriever(user_role)(clean_input)
        context = "\n\n".join(doc.page_content for doc in docs)

        chain = RunnableParallel(context=lambda _: context, question=RunnablePassthrough()) | secure_prompt | _llm | StrOutputParser()
        answer = chain.invoke(clean_input)

        answer = pre_filter_check(answer)
        answer = model_guard_check(answer, context)

        confidence = "HIGH" if len(answer.split()) > 10 and docs else "LOW"

        audit.log("SUCCESS", {"confidence": confidence, "answer_length": len(answer)})
        return {"answer": answer, "confidence": confidence}

    except Exception as e:
        audit.log("BLOCKED", {"error": str(e)})
        return {
            "answer": "I'm sorry, I cannot assist with that request due to security policy.",
            "confidence": "BLOCKED"
        }