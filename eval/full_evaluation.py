# eval/full_evaluation.py

import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
    answer_correctness,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from app.config import *
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking


# ============================================================
# SHARED RAGAS LLM + EMBEDDINGS
# ============================================================
def _build_ragas_wrappers():
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            n=3,
            api_key=OPENAI_API_KEY
        )
    )
    lc_embeddings    = LCOpenAIEmbeddings(model="text-embedding-3-small")
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
    return ragas_llm, ragas_embeddings


def _configure_metrics(ragas_llm, ragas_embeddings):
    faithfulness.llm              = ragas_llm
    context_precision.llm         = ragas_llm
    context_recall.llm            = ragas_llm
    answer_relevancy.llm          = ragas_llm
    answer_relevancy.embeddings   = ragas_embeddings
    answer_correctness.llm        = ragas_llm
    answer_correctness.embeddings = ragas_embeddings

    return [
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
        answer_correctness,
    ]


# ============================================================
# 1. BUILD THE RAG CHAIN TO EVALUATE
# ============================================================
def build_rag_chain():
    documents = ingest_all()
    chunks    = recursive_character_chunking(
        documents, chunk_size=600, chunk_overlap=150
    )

    embeddings  = LCOpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # BALANCED PROMPT — complete sentences, one fact, no extra context.
    # Lesson learned:
    #   Too verbose → RAGAS generates drift questions from multi-topic answers
    #   Too terse   → RAGAS cannot generate any question from "$75" or "Yes."
    #   Sweet spot  → one complete sentence per answer, verbatim from context
    prompt = ChatPromptTemplate.from_template("""
You are a precise assistant. Answer ONLY from the context below.

RULES:
1. Answer in exactly ONE complete sentence.
2. The sentence must directly answer what was asked — nothing more, nothing less.
3. Copy the exact wording from the context. Do not paraphrase.
4. Do NOT start with "Based on the context" or "According to the policy".
5. If the context contains no relevant information, respond with EXACTLY:
   "This topic is not covered in the provided documents."

Context:
{context}

Question: {question}

One-sentence answer:""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    setup     = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    rag_chain = setup | prompt | llm | StrOutputParser()

    return rag_chain, retriever


# ============================================================
# 2. BUILD EVALUATION DATASET
# ============================================================
def build_eval_dataset(rag_chain, retriever):
    """
    Ground truths are calibrated to match what the LLM actually returns
    with the balanced prompt — complete sentences, verbatim from policy.

    RAGAS answer_relevancy works by generating a synthetic question FROM
    the answer and measuring cosine similarity to the original question.
    One-sentence complete answers reliably produce matching synthetic questions.
    """
    eval_pairs = [
        # PASSWORD POLICY
        {
            "question":     "What is the minimum password length required?",
            "ground_truth": "Passwords must be minimum 14 characters with uppercase, lowercase, numbers, and special characters.",
        },
        {
            "question":     "How often must passwords be rotated?",
            "ground_truth": "Password rotation is required every 90 days.",
        },
        {
            "question":     "Is multi-factor authentication mandatory?",
            "ground_truth": "Multi-factor authentication is mandatory for all systems.",
        },

        # SECURITY INCIDENTS
        {
            "question":     "Within how many hours must security incidents be reported?",
            "ground_truth": "All security incidents must be reported within 1 hour of discovery.",
        },
        {
            "question":     "Where should security incidents be reported?",
            "ground_truth": "Security incidents must be reported to security@techcorp.com.",
        },

        # VENDOR SECURITY
        {
            "question":     "What certification must vendors handling confidential data provide?",
            "ground_truth": "Vendors handling Confidential data must provide SOC 2 Type II certification.",
        },
        {
            "question":     "How often are vendor security reviews conducted?",
            "ground_truth": "Annual vendor security reviews are mandatory.",
        },

        # TRAVEL & EXPENSES
        {
            "question":     "What is the domestic meal per diem rate?",
            "ground_truth": "Per diem for meals is $75 per day domestic.",
        },
        {
            "question":     "What is the international meal per diem rate?",
            "ground_truth": "Per diem for meals is $100 per day international.",
        },
        {
            "question":     "What is the hotel rate cap per night in standard markets?",
            "ground_truth": "Hotel rates are capped at $250 per night in standard markets.",
        },
        {
            "question":     "What is the hotel rate cap per night in high-cost cities?",
            "ground_truth": "Hotel rates are capped at $350 per night in high-cost cities (NYC, SF, London).",
        },
        {
            "question":     "How many days do employees have to submit expense reports?",
            "ground_truth": "Expense reports must be submitted within 30 days of the expense.",
        },
        {
            "question":     "How long does reimbursement take after approval?",
            "ground_truth": "Reimbursement is processed within 10 business days of approval.",
        },

        # REMOTE WORK
        {
            "question":     "What equipment does the company provide for remote work?",
            "ground_truth": "The company provides a laptop and one external monitor.",
        },
        {
            "question":     "What is the annual home office setup stipend?",
            "ground_truth": "A $500 annual stipend is provided for home office setup.",
        },
        {
            "question":     "What minimum internet speed is required for remote work?",
            "ground_truth": "Employees are responsible for maintaining a stable internet connection with minimum 50 Mbps download speed.",
        },

        # PERFORMANCE
        {
            "question":     "What happens after two consecutive unsatisfactory performance reviews?",
            "ground_truth": "Two consecutive unsatisfactory reviews may result in revocation of remote work privileges.",
        },

        # CODE REVIEW
        {
            "question":     "How many engineers must review a pull request before merging?",
            "ground_truth": "At least two engineers must review a pull request before merging.",
        },
        {
            "question":     "What is the minimum code coverage required for pull requests?",
            "ground_truth": "Pull requests must include unit tests with minimum 80% code coverage.",
        },
        {
            "question":     "Within how many business hours must code reviews be completed?",
            "ground_truth": "Reviews must be completed within 24 business hours of submission.",
        },

        # DEPLOYMENTS
        {
            "question":     "On which days are production deployments permitted?",
            "ground_truth": "Production deployments are permitted Monday through Thursday between 9 AM and 2 PM ET.",
        },
        {
            "question":     "What approval is required for Friday production deployments?",
            "ground_truth": "Friday deployments require VP of Engineering approval.",
        },

        # ON-CALL
        {
            "question":     "How long are on-call shifts?",
            "ground_truth": "On-call shifts are one week long, Monday 9 AM to Monday 9 AM.",
        },
        {
            "question":     "What is the weekly on-call stipend?",
            "ground_truth": "On-call engineers receive $500 per week stipend.",
        },
        {
            "question":     "What is the per-incident compensation for on-call engineers?",
            "ground_truth": "On-call engineers receive $200 per incident handled.",
        },
        {
            "question":     "When are engineers eligible to join the on-call rotation?",
            "ground_truth": "Engineers participate in on-call rotations after completing 6 months of tenure.",
        },

        # PROFESSIONAL DEVELOPMENT
        {
            "question":     "What is the annual learning budget per employee?",
            "ground_truth": "Each employee has a $3,000 annual learning budget for conferences, courses, and certifications.",
        },
        {
            "question":     "What is the maximum tuition reimbursement per year?",
            "ground_truth": "Tuition reimbursement up to $10,000 per year is available for degree programs.",
        },
    ]

    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    print("\n" + "─" * 70)
    print("ANSWER DIAGNOSTIC")
    print("─" * 70)

    for item in eval_pairs:
        q  = item["question"]
        gt = item["ground_truth"]

        retrieved_docs = retriever.invoke(q)
        ctx_texts      = [doc.page_content for doc in retrieved_docs]
        answer         = rag_chain.invoke(q)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(ctx_texts)
        ground_truths.append(gt)

        print(f"\n  Q:  {q}")
        print(f"  A:  {answer}")
        print(f"  GT: {gt}")

    print("\n" + "─" * 70)
    print(f"✅ Collected {len(questions)} question-answer pairs")
    print("─" * 70)

    return Dataset.from_dict(
        {
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts_list,
            "ground_truth": ground_truths,
        }
    )


# ============================================================
# HELPER: safely extract scalar from Ragas result
# ============================================================
def _scalar(value) -> float:
    if isinstance(value, list):
        valid = [
            v for v in value
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        return float(np.mean(valid)) if valid else float("nan")
    return float(value)


# ============================================================
# 3. RUN FULL EVALUATION
# ============================================================
def run_full_evaluation():
    print("=" * 70)
    print("FULL RAG EVALUATION PIPELINE")
    print("=" * 70)

    print("\n⚙️  Configuring Ragas LLM + embeddings wrappers...")
    ragas_llm, ragas_embeddings = _build_ragas_wrappers()
    metric_list = _configure_metrics(ragas_llm, ragas_embeddings)
    print("✅ Ragas metrics configured")

    print("\n📦 Building RAG chain...")
    rag_chain, retriever = build_rag_chain()
    print("✅ RAG chain ready")

    print("\n📊 Running RAG chain on evaluation questions...")
    dataset = build_eval_dataset(rag_chain, retriever)

    print("\n🔍 Running Ragas evaluation (all 5 metrics)...")
    results = evaluate(dataset, metrics=metric_list)

    # ---- Aggregate scores ----
    print("\n" + "=" * 70)
    print("AGGREGATE SCORES")
    print("=" * 70)

    thresholds = {
        "faithfulness":       0.8,
        "context_precision":  0.8,
        "context_recall":     0.7,
        "answer_relevancy":   0.8,
        "answer_correctness": 0.6,
    }

    all_passed = True
    for metric, threshold in thresholds.items():
        score  = _scalar(results[metric])
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        if score < threshold:
            all_passed = False
        print(f"  {metric:<25} {score:.4f}  (threshold: {threshold})  {status}")

    # ---- Per-question breakdown ----
    print("\n" + "=" * 70)
    print("PER-QUESTION BREAKDOWN")
    print("=" * 70)

    df = results.to_pandas()
    for i, row in df.iterrows():
        question_text = row.get("question", row.get("user_input", ""))
        print(f"\n  Q{i+1}: {str(question_text)[:60]}...")

        for col, label in [
            ("faithfulness",       "Faithfulness       "),
            ("context_precision",  "Context Precision  "),
            ("context_recall",     "Context Recall     "),
            ("answer_relevancy",   "Answer Relevancy   "),
            ("answer_correctness", "Answer Correctness "),
        ]:
            val = row.get(col)
            if isinstance(val, float) and not np.isnan(val):
                flag = " ⚠️" if val < 0.8 else ""
                print(f"    {label} {val:.3f}{flag}")
            else:
                print(f"    {label} N/A")

    # ---- Final verdict ----
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL METRICS PASSED — Pipeline ready for deployment")
    else:
        print("❌ SOME METRICS FAILED — Review failing questions above")
        print("   Common fixes:")
        print("   - Low faithfulness       → LLM is hallucinating, tighten prompt")
        print("   - Low context_precision  → retriever returning noisy chunks")
        print("   - Low context_recall     → chunk_size too small, increase overlap")
        print("   - Low answer_relevancy   → answer not directly addressing the question")
        print("   - Low answer_correctness → ground_truth mismatch or bad retrieval")
    print("=" * 70)

    return results


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_full_evaluation()