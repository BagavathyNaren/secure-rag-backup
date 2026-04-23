# eval/fine_tuning.py

import sys
import os
import json
import time
from pydantic import SecretStr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import *
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ============================================================
# PART 1: SYNTHETIC TRAINING DATA GENERATION
# ============================================================
def generate_training_data():
    """
    Uses the RAG pipeline to generate synthetic Q&A pairs
    formatted for OpenAI fine-tuning.
    """
    # --- Setup LLM ---
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
    )

    # --- Load and chunk documents ---
    print("  Loading documents...")
    documents = ingest_all()
    print(f"  Loaded {len(documents)} documents.")

    print("  Chunking documents...")
    chunks = recursive_character_chunking(documents)
    print(f"  Total chunks: {len(chunks)}")

  

     # Balanced filter: Core policy files + engineering standards (contains PR/on-call rules)
    allowed_sources = [
        "company_policy.txt",
        "employee_handbook.pdf",
        "engineering_standards.docx"
    ]
    
    usable_chunks = [
        c for c in chunks 
        if len(c.page_content) >= 100 
        and any(src in str(c.metadata.get('source', '')) for src in allowed_sources)
    ]
    print(f"  Usable HR/policy chunks: {len(usable_chunks)}")

    # --- Prompt + chain ---
    prompt = ChatPromptTemplate.from_template(
        """Given this policy excerpt, generate ONE specific question and its exact one-sentence answer.

Rules:
- The answer must be a complete sentence that ends with a period.
- Use 8 or more words in the answer.
- The answer must be a direct quote or very close paraphrase from the excerpt.
- Never output key:value format, bullet points, or just numbers.
- Output ONLY valid JSON with no extra text: {{"question": "...", "answer": "..."}}

Excerpt: {chunk_text}"""
    )

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # --- System prompt used for all training examples ---
    system_prompt = (
        "You are a precise HR assistant for TechCorp. "
        "Answer in exactly one sentence using exact policy wording."
    )

    training_examples = []
    failed = 0

    print(f"\n  Generating Q&A pairs from {len(usable_chunks)} chunks...")

    for i, chunk in enumerate(usable_chunks):
        chunk_text = chunk.page_content.strip()

        try:
            result = chain.invoke({"chunk_text": chunk_text})

            # Validate: result must have non-empty question and answer
            question = result.get("question", "").strip()
            answer = result.get("answer", "").strip()

            # === NEW STRICT FILTERING ===
            if not question or not answer:
                failed += 1
                continue
       
             # Softer but still effective filters
            if not answer.strip().endswith('.'):
                answer = answer.strip() + "."
                
            if len(answer.split()) < 6 or ':' in answer[:20]:
                failed += 1
                continue

            # Format as OpenAI fine-tuning JSONL entry
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            training_examples.append(entry)

        except Exception as e:
            failed += 1
            # Silently skip bad chunks — don't pollute training data
            continue

        # Progress update every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(usable_chunks)} chunks | "
                  f"Generated: {len(training_examples)} | Failed: {failed}")

        # Early stop once we have enough (avoids excessive API cost)
        if len(training_examples) >= 100:
            print(f"  Reached 100 examples — stopping early.")
            break

    print(f"\n  ✅ Generated {len(training_examples)} training examples "
          f"({failed} chunks skipped).")

    if len(training_examples) < 10:
        raise ValueError(
            f"Only {len(training_examples)} examples generated. "
            "OpenAI requires minimum 10. Check your documents."
        )

    # --- Save to JSONL ---
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_data.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in training_examples:
            f.write(json.dumps(entry) + "\n")

    print(f"  💾 Saved to {output_path}")
    return output_path


# ============================================================
# PART 2: OPENAI FINE-TUNING
# ============================================================
def upload_and_finetune():
    """
    Uploads training data to OpenAI and starts a fine-tuning job.
    Polls until completion and returns the fine-tuned model ID.
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    training_file_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "training_data.jsonl"
    )

    if not os.path.exists(training_file_path):
        raise FileNotFoundError(
            f"Training data not found at {training_file_path}. "
            "Run generate_training_data() first."
        )

    # --- Step 1: Upload file ---
    print(f"  Uploading {training_file_path} to OpenAI Files API...")
    with open(training_file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id
    print(f"  ✅ Uploaded file: {file_id}")

    # --- Step 2: Create fine-tuning job ---
    print("  Creating fine-tuning job on gpt-4o-mini...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",   # exact snapshot required by OpenAI
        hyperparameters={"n_epochs": 3},
        suffix="techcorp-rag",
    )
    job_id = job.id
    print(f"  ✅ Fine-tuning job started: {job_id}")
    print("  ⏳ Training typically takes 10–30 minutes. Polling every 30 seconds...\n")

    # --- Step 3: Poll until done ---
    poll_count = 0
    while True:
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        status = job_status.status
        poll_count += 1
        print(f"  [{poll_count:03d}] Status: {status}")

        if status in ("succeeded", "failed", "cancelled"):
            break

        time.sleep(30)

    # --- Step 4: Return model ID ---
    if job_status.status == "succeeded":
        model_id = job_status.fine_tuned_model
        print(f"\n  ✅ Fine-tuned model ready: {model_id}")
        return model_id
    else:
        print(f"\n  ❌ Fine-tuning ended with status: {job_status.status}")
        return None


# ============================================================
# PART 3: COMPARE BASE vs FINE-TUNED
# ============================================================
def compare_base_vs_finetuned(finetuned_model_id: str):
    """
    Runs the same 5 test questions against:
    - gpt-4o-mini (base)
    - your fine-tuned model

    Compares: response format, answer precision, instruction following.
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a precise HR assistant for TechCorp. "
        "Answer in exactly one sentence using exact policy wording."
    )

    test_questions = [
        "What is the minimum password length?",
        "What is the weekly on-call stipend?",
        "How many engineers must review a PR?",
        "What is the hotel cap in high-cost cities?",
        "When can engineers join on-call rotation?",
    ]

    def ask(model: str, question: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    print("\n" + "=" * 70)
    print("BASE vs FINE-TUNED COMPARISON")
    print("=" * 70)
    print(f"  Base model:       gpt-4o-mini")
    print(f"  Fine-tuned model: {finetuned_model_id}")
    print("=" * 70)

    results = []

    for q in test_questions:
        print(f"\n📝 {q}")
        base_answer = ask("gpt-4o-mini", q)
        ft_answer = ask(finetuned_model_id, q)
        print(f"  Base:        {base_answer}")
        print(f"  Fine-Tuned:  {ft_answer}")

        results.append({
            "question": q,
            "base": base_answer,
            "finetuned": ft_answer,
        })

    # --- Simple format analysis ---
    print("\n" + "=" * 70)
    print("FORMAT ANALYSIS")
    print("=" * 70)

    base_avg_len = sum(len(r["base"]) for r in results) / len(results)
    ft_avg_len = sum(len(r["finetuned"]) for r in results) / len(results)

    # Count how many answers end with a period (complete sentence signal)
    base_complete = sum(1 for r in results if r["base"].strip().endswith("."))
    ft_complete = sum(1 for r in results if r["finetuned"].strip().endswith("."))

    print(f"  Average response length — Base: {base_avg_len:.0f} chars | "
          f"Fine-tuned: {ft_avg_len:.0f} chars")
    print(f"  Ends with period        — Base: {base_complete}/5 | "
          f"Fine-tuned: {ft_complete}/5")

    return results


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":

    print("=" * 70)
    print("RAG FINE-TUNING PIPELINE")
    print("=" * 70)

    # Part 1: Generate training data
    print("\n📝 PART 1: Generating synthetic training data...")
    training_file = generate_training_data()
    # training_file = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.jsonl")
    # print(f"Using existing training file: {training_file}")

    # Part 2: Upload and fine-tune
    print("\n🚀 PART 2: Uploading to OpenAI and starting fine-tune...")
    finetuned_model_id = upload_and_finetune()

    # Part 3: Compare
    if finetuned_model_id:
        print("\n🔍 PART 3: Comparing base vs fine-tuned model...")
        compare_base_vs_finetuned(finetuned_model_id)
    else:
        print("\n⚠️  Skipping comparison — fine-tuning did not succeed.")

    print("\n" + "=" * 70)
    print("FINE-TUNING PIPELINE COMPLETE")
    print("=" * 70)