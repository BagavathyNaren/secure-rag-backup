# app/embeddings_comparison.py

import time
from typing import List
from langchain_core.documents import Document
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking
from app.config import *

# ============================================================
# 1. EMBEDDING MODELS
# ============================================================

def get_openai_embeddings():
    """
    Returns OpenAI embedding model instance.
    Model: text-embedding-3-small → 1536 dimensions
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

def get_huggingface_embeddings():
    """
    Returns a FREE local HuggingFace embedding model.
    Model: all-MiniLM-L6-v2 → 384 dimensions
    First run downloads ~80MB. Subsequent runs use cache.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},       # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )


# ============================================================
# 2. VECTOR STORE BUILDERS
# ============================================================

def build_faiss_store(chunks: List[Document], embeddings) -> tuple:
    """
    Build a FAISS in-memory vector store.
    Returns (vectorstore, build_time_seconds).
    """
    from langchain_community.vectorstores import FAISS

    start = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start

    return vectorstore, elapsed


def build_chroma_store(
    chunks: List[Document], embeddings, collection_name: str
) -> tuple:
    """
    Build a Chroma vector store.
    Returns (vectorstore, build_time_seconds).

    collection_name must be unique per embedding model to avoid
    dimension mismatches inside the same Chroma instance.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_community.vectorstores.utils import filter_complex_metadata

    # Strip nested dicts/tuples that Chroma can't store
    filtered_chunks = filter_complex_metadata(chunks)

    start = time.time()
    vectorstore = Chroma.from_documents(
        filtered_chunks,
        embeddings,
        collection_name=collection_name,
    )
    elapsed = time.time() - start

    return vectorstore, elapsed


def build_qdrant_store(
    chunks: List[Document], embeddings, collection_name: str
) -> tuple:
    """
    Build a Qdrant vector store against the local Docker container.
    Returns (vectorstore, build_time_seconds).

    Prerequisites:
        docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
    
    force_recreate=True wipes and rebuilds the collection on every
    run — safe for experiments, don't use in production.
    """
    from langchain_qdrant import QdrantVectorStore
    from langchain_community.vectorstores.utils import filter_complex_metadata
    from qdrant_client import QdrantClient, models

    
    # Strip nested dicts/tuples that Chroma can't store
    filtered_chunks = filter_complex_metadata(chunks)
    # Explicit timeout prevents ReadTimeout on collection create/recreate
    client = QdrantClient(host="localhost", port=6333, timeout=60)

    start = time.time()
    vectorstore = QdrantVectorStore.from_documents(
        filtered_chunks,
        embeddings,
        url="http://localhost:6333",
        collection_name=collection_name,
        force_recreate=True,
        timeout=60
    )
    elapsed = time.time() - start

    return vectorstore, elapsed


def build_pinecone_store(
    chunks: List[Document], embeddings, index_name: str
) -> tuple:
    """
    Build a Pinecone vector store on the free-tier cloud.
    Returns (vectorstore, build_time_seconds).

    Dimension is auto-detected from the embedding model so the same
    function works for both OpenAI (1536) and HuggingFace (384).

    Free tier limit: 5 indexes.  Use descriptive names:
        "openai-index"   → 1536 dims
        "hf-index"       → 384  dims
    """
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    import time as t
    from langchain_community.vectorstores.utils import filter_complex_metadata

    
    # Strip nested dicts/tuples that Chroma can't store
    filtered_chunks = filter_complex_metadata(chunks)

    # --- detect embedding dimension from a probe vector ---
    probe = embeddings.embed_query("probe")
    dimension = len(probe)

    # --- init client & create index if needed ---
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Poll until ready (usually < 60s on free tier)
        while not pc.describe_index(index_name).status["ready"]:
            t.sleep(1)
    else:
        # Index exists — verify dimension matches to catch misuse early
        existing_dim = pc.describe_index(index_name).dimension
        if existing_dim != dimension:
            raise ValueError(
                f"Pinecone index '{index_name}' has dimension {existing_dim} "
                f"but embedding model produces {dimension} dims. "
                f"Delete the index in the Pinecone dashboard and re-run."
            )

    start = time.time()
    vectorstore = PineconeVectorStore.from_documents(
        filtered_chunks,
        embeddings,
        index_name=index_name,
    )
    elapsed = time.time() - start

    return vectorstore, elapsed


# ============================================================
# 3. QUERY BENCHMARK
# ============================================================

def benchmark_retrieval(vectorstore, query: str, k: int = 3) -> dict:
    """
    Run a similarity search and return results + timing.

    Returns:
        {
            "results":     [first 100 chars of each retrieved doc],
            "time":        retrieval_time_in_seconds,
            "num_results": number of docs returned
        }
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    start = time.time()
    docs = retriever.invoke(query)
    elapsed = time.time() - start

    return {
        "results": [doc.page_content[:300] for doc in docs],
        "time": elapsed,
        "num_results": len(docs),
    }


# ============================================================
# 4. FULL COMPARISON
# ============================================================

def run_comparison():
    """
    Complete comparison across 4 vector DBs × 2 embedding models.
    """

    # ----------------------------------------------------------
    # Step 1: Load and chunk documents
    # ----------------------------------------------------------
    print("Loading and chunking documents...")
    docs = ingest_all()
    chunks = recursive_character_chunking(docs, chunk_size=500, chunk_overlap=100)
    print(f"\nTotal chunks to embed: {len(chunks)}")

    # ----------------------------------------------------------
    # Step 2: Initialize embedding models
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("INITIALIZING EMBEDDING MODELS")
    print("=" * 70)

    openai_emb = get_openai_embeddings()
    print("✅ OpenAI embeddings ready (dim: 1536)")

    hf_emb = get_huggingface_embeddings()
    print("✅ HuggingFace embeddings ready (dim: 384)")

    # ----------------------------------------------------------
    # Step 3: Build all 8 vector store combinations
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING VECTOR STORES (8 combinations)")
    print("=" * 70)

    stores = {}

    # FAISS — pure in-memory, no external service needed
    store, t = build_faiss_store(chunks, openai_emb)
    stores["FAISS + OpenAI"] = store
    print(f"  ✅ FAISS + OpenAI:          {t:.2f}s")

    store, t = build_faiss_store(chunks, hf_emb)
    stores["FAISS + HuggingFace"] = store
    print(f"  ✅ FAISS + HuggingFace:     {t:.2f}s")

    # Chroma — local persistent store
    store, t = build_chroma_store(chunks, openai_emb, "openai_col")
    stores["Chroma + OpenAI"] = store
    print(f"  ✅ Chroma + OpenAI:         {t:.2f}s")

    store, t = build_chroma_store(chunks, hf_emb, "hf_col")
    stores["Chroma + HuggingFace"] = store
    print(f"  ✅ Chroma + HuggingFace:    {t:.2f}s")

    # Qdrant — local Docker
    store, t = build_qdrant_store(chunks, openai_emb, "openai_qdrant")
    stores["Qdrant + OpenAI"] = store
    print(f"  ✅ Qdrant + OpenAI:         {t:.2f}s")

    store, t = build_qdrant_store(chunks, hf_emb, "hf_qdrant")
    stores["Qdrant + HuggingFace"] = store
    print(f"  ✅ Qdrant + HuggingFace:    {t:.2f}s")

    # Pinecone — cloud, dimension auto-detected per embedding model
    store, t = build_pinecone_store(chunks, openai_emb, "openai-index")
    stores["Pinecone + OpenAI"] = store
    print(f"  ✅ Pinecone + OpenAI:       {t:.2f}s")

    store, t = build_pinecone_store(chunks, hf_emb, "hf-index")
    stores["Pinecone + HuggingFace"] = store
    print(f"  ✅ Pinecone + HuggingFace:  {t:.2f}s")

    # ----------------------------------------------------------
    # Step 4: Run benchmark queries
    # ----------------------------------------------------------
    test_queries = [
        "What is the minimum password length requirement?",
        "How much is the travel per diem for international trips?",
        "What are the code review requirements?",
        "What is the on-call stipend amount?",
        "What is the reimbursement amount for degree programs?",
        "What are the 'Payment Terms' mentioned in the invoice 'INV-2025-001'?",
            # CSV — sales_data.csv
        "What was the revenue for North America in Q1 2025?",
        "Under the Asia-Pacific region, who is the sales rep for Laptop Pro 15 in Q2 2025?",
              # CSV — indian_sales_data.csv
        "What was the number of units sold for Mumbai in Q1 2026?",
        "Who is the sales rep for Lenovo Legion in Q4 2025?",
       # Excel — infrastructure_inventory.xlsx
       "What servers are running in us-east-1?",
       "How many CPU cores does prod-api-01 have?",
       "What is the license type for software 'Figma'?",
       "What is the SLA for the Vendor 'DataPipe Analytics'?",
       "What is the response time for the 'SEV-2'?",

    ]

    print("\n" + "=" * 70)
    print("RETRIEVAL BENCHMARK")
    print("=" * 70)

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 60)

        for store_name, store in stores.items():
            result = benchmark_retrieval(store, query)
            print(f"\n  [{store_name}] ({result['time']:.4f}s)")
            for i, doc_text in enumerate(result["results"]):
                print(f"    Result {i+1}: {doc_text}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_comparison()