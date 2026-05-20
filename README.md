---
title: Secure RAG API
emoji: 🔐
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TechCorp Secure RAG API

Enterprise RAG API with JWT authentication, role-based retrieval, prompt-injection checks, PII redaction, output guardrails, cache support, and FAISS/PGVector backends.

## Required Secrets

Set these in Hugging Face Space secrets before deploying:

- `OPENAI_API_KEY`
- `DATABASE_URL`
- `JWT_SECRET_KEY` (random value, at least 32 characters)

Optional but recommended:

- `CORS_ALLOW_ORIGINS` for browser frontends
- `EVAL_API_KEY` for internal/evaluation endpoints
- `REDIS_URL` for shared response cache

## Authentication

`POST /auth/login`

Use form data:

```text
username=alice
password=<configured seed password>
```

Response:

```json
{
  "access_token": "...",
  "token_type": "bearer",
  "expires_in_minutes": 60,
  "role": "employee"
}
```

## RAG Endpoint

`POST /secure-rag/invoke`

The role comes from the JWT, not from request JSON.

Headers:

```text
Authorization: Bearer <access_token>
Content-Type: application/json
```

Request:

```json
{
  "question": "What is the minimum password length?"
}
```

Response is Server-Sent Events:

```text
data: {"token":"The "}
data: {"token":"minimum "}
data: {"done":true,"answer":"...","confidence":"HIGH","cached":false}
```

## Health

- `GET /health` is a cheap liveness check and does not call OpenAI.
- `GET /ready` checks database connectivity and initialized local components.

## Demo Users

Demo seeding is disabled by default. To seed users, set `SEED_DEMO_USERS=1` and provide per-user password secrets such as `SEED_ALICE_PASSWORD`, `SEED_FRANK_PASSWORD`, and `SEED_GRACE_PASSWORD`. No default demo passwords are shipped.

## Local Smoke Test

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export DATABASE_URL=...
export JWT_SECRET_KEY=$(openssl rand -hex 32)
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

Then login and call `/secure-rag/invoke` with the returned bearer token.
