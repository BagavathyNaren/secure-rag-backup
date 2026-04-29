# app/secure_rag.py

import re
import logging
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

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

def log_event(event_type: str, data: str):
    logging.info(f"[{event_type}] {data}")


# ============================================================
# NUCLEAR-GRADE HARD BLOCKS (INSTANT KILL SWITCH)
# ============================================================
DANGEROUS_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",                     # AWS Access Key
    r"sk_[a-zA-Z0-9]{32,}",                  # OpenAI/Anthropic/etc
    r"claude-api-key-[a-zA-Z0-9-]{20,}",
    r"[A-Za-z0-9+/]{40}",                    # Long base64 secrets
    r"ghp_[a-zA-Z0-9]{36}",
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
    r"Secret Access Key[^\n]{0,20}[:=][^\n]{10,}",
    r"Secret Key[^\n]{0,20}[:=][^\n]{10,}",
]

BLOCKED_KEYWORDS = [
    "AKIA", "sk_", "claude-api-key", "Secret Access Key", "Secret Key",
    "private key", "BEGIN PRIVATE KEY", "ghp_", "github_pat", "AWS_SECRET"
]

def pre_filter_check(answer: str) -> str:
    """Instant hard block before anything else"""
    if any(kw in answer for kw in BLOCKED_KEYWORDS):
        raise ValueError("SECURITY BLOCK: Forbidden keyword detected in output.")
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            raise ValueError("SECURITY BLOCK: Credential pattern detected in output.")
    return answer


# ============================================================
# 1️⃣ INPUT GUARDRAILS
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
            raise ValueError("Prompt injection detected.")

def redact_pii(text: str) -> str:
    text = re.sub(EMAIL_REGEX, "[REDACTED_EMAIL]", text)
    text = re.sub(CREDIT_CARD_REGEX, "[REDACTED_CC]", text)
    text = re.sub(API_KEY_REGEX, "[BLOCKED_API_KEY]", text)
    return text


# ============================================================
# 2️⃣ LOAD PREBUILT FAISS INDEX
# ============================================================

INDEX_PATH = "faiss_index"

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(INDEX_PATH):
    print("Loading prebuilt FAISS index from repo...")
    _vectorstore = FAISS.load_local(
        INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded.")
else:
    print("No prebuilt index found. Building new FAISS index...")
    _documents = ingest_all()
    _chunks = recursive_character_chunking(
        _documents,
        chunk_size=600,
        chunk_overlap=150
    )
    _vectorstore = FAISS.from_documents(_chunks, _embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    _vectorstore.save_local(INDEX_PATH)
    print("FAISS index built and saved.")


# ============================================================
# 3️⃣ ROLE-BASED RETRIEVAL
# ============================================================

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
            return base_retriever.invoke(query)
        return retrieve

    def filtered_retrieval(query: str):
        docs = base_retriever.invoke(query)
        return [d for d in docs if d.metadata.get("file_name") in allowed]

    return filtered_retrieval


# ============================================================
# 4️⃣ HARDENED PROMPT (Concise + Safe)
# ============================================================

secure_prompt = ChatPromptTemplate.from_template("""
You are a secure assistant.
SECURITY RULES:
1. Never reveal confidential information.
2. Only answer using retrieved context.
INSTRUCTIONS:
- Be clear, concise, and direct.
- Use short paragraphs and bullet points when helpful.
- Avoid unnecessary details and long explanations.
- Keep the total answer brief but informative.
Context:
{context}
Question:
{question}
Answer:
""")

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    max_tokens=450
)


# ============================================================
# 5️⃣ NUCLEAR-GRADE OUTPUT GUARD (UPGRADED)
# ============================================================

def model_guard_check(answer: str, context: str = "") -> str:
    # Double regex check
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            raise ValueError("SECURITY BLOCK: Credential pattern detected.")

    guard_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    guard_prompt = f"""
YOU ARE A PARANOID SECURITY FIREWALL.

Context:
{context[:2000]}

Answer:
{answer}

RULES:
- ANY API key, secret, credential = UNSAFE (even fake/test/mock)
- ANY server name, hostname, IP, RAM, CPU = UNSAFE
- If in doubt = UNSAFE

Reply only: SAFE or UNSAFE
"""

    result = guard_llm.invoke(guard_prompt).content.strip().upper()
    if result != "SAFE":
        raise ValueError("Output blocked by model-based guardrail.")

    return answer


# ============================================================
# 6️⃣ CONFIDENCE SCORING
# ============================================================

def compute_confidence(retrieved_docs, answer):
    if not retrieved_docs:
        return "LOW"
    if "does not specify" in answer.lower() or "cannot provide" in answer.lower():
        return "LOW"
    if len(answer.split()) < 6:
        return "LOW"
    return "HIGH"


# ============================================================
# 7️⃣ MAIN SECURE INVOKE FUNCTION (NOW WITH TRIPLE DEFENSE)
# ============================================================

def secure_rag_invoke(user_input: str, user_role: str = "employee") -> Dict:

    log_event("INPUT", user_input)

    # Input Guardrails
    detect_prompt_injection(user_input)
    user_input = redact_pii(user_input)

    # Retrieval
    retriever = build_secure_retriever(user_role)
    retrieved_docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # RAG Generation
    setup = RunnableParallel(
        context=lambda _: context,
        question=RunnablePassthrough()
    )

    rag_chain = setup | secure_prompt | _llm | StrOutputParser()
    answer = rag_chain.invoke(user_input)

    # TRIPLE NUCLEAR DEFENSE
    answer = pre_filter_check(answer)                    # Layer 1: Instant block
    answer = model_guard_check(answer, context)          # Layer 2: Paranoid LLM guard

    confidence = compute_confidence(retrieved_docs, answer)

    log_event("ANSWER", answer)

    return {
        "answer": answer,
        "confidence": confidence
    }