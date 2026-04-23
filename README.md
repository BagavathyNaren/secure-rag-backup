---
title: Secure RAG API
emoji: 🔐
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TechCorp Secure RAG API

Enterprise-grade RAG system with:

- Multi-source document ingestion (PDF, DOCX, Excel, CSV, Web)
- RBAC-based retrieval filtering
- Prompt injection detection
- PII redaction
- Model-based output guardrails
- Confidence scoring
- Persistent FAISS indexing

## Endpoint

POST /secure-rag/invoke

## Request

\`\`\`json
{
  "question": "What is the minimum password length?",
  "role": "employee"
}
\`\`\`

## Response

\`\`\`json
{
  "answer": "The minimum password length is 14 characters.",
  "confidence": "HIGH"
}
\`\`\`