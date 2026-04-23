# app/advanced_retrieval.py

import time
from typing import List
from langchain_core.documents import Document
from app.config import *
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking

# ============================================================
# SETUP: Build base vector store
# ============================================================
def build_base_vectorstore():
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    docs = ingest_all()
    chunks = recursive_character_chunking(docs, chunk_size=500, chunk_overlap=100)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore, chunks

# ============================================================
# STRATEGY 1: MULTI-QUERY RETRIEVAL
# ============================================================
def create_multi_query_retriever(vectorstore):
    # USE DEEP PATH: langchain.retrievers.multi_query
    from langchain_classic.retrievers import MultiQueryRetriever
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )
    return retriever

# ============================================================
# STRATEGY 2: CONTEXTUAL COMPRESSION
# ============================================================
def create_compressed_retriever(vectorstore):
    # USE DEEP PATH: langchain.retrievers.contextual_compression
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    return retriever

# ============================================================
# STRATEGY 3: ENSEMBLE RETRIEVAL (BM25 + Vector)
# ============================================================
def create_ensemble_retriever(vectorstore, chunks: List[Document]):
    # USE DEEP PATHS
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    bm25_retriever = BM25Retriever.from_documents(chunks, k=3)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )
    return retriever

# ============================================================
# STRATEGY 4: CROSS-ENCODER RE-RANKING
# ============================================================
def create_reranking_retriever(vectorstore):
    # USE DEEP PATHS
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
    )
    return retriever


# ============================================================
# COMPARISON
# ============================================================
def compare_retrieval_strategies():
    print("Building base vector store...")
    vectorstore, chunks = build_base_vectorstore()

    print("Initializing retrieval strategies...")
    strategies = {
        "1. Base Vector (k=3)": vectorstore.as_retriever(search_kwargs={"k": 3}),
        "2. Multi-Query": create_multi_query_retriever(vectorstore),
        "3. Contextual Compression": create_compressed_retriever(vectorstore),
        "4. Ensemble BM25+Vector": create_ensemble_retriever(vectorstore, chunks),
        "5. Cross-Encoder Reranked": create_reranking_retriever(vectorstore),
    }
    print("✅ All strategies ready\n")

    test_queries = [
        "What is the minimum password length?",
        "How do employees get reimbursed for travel?",
        "SEV-1 incident response time",
        "What certifications do vendors need?",
    ]

    print("=" * 70)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 70)

    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"📝 Query: '{query}'")
        print(f"{'─' * 70}")

        for name, retriever in strategies.items():
            start = time.time()
            try:
                results = retriever.invoke(query)
            except Exception as e:
                print(f"\n  [{name}] ERROR: {e}")
                continue
            elapsed = time.time() - start

            print(f"\n  [{name}] ({elapsed:.3f}s) — {len(results)} results")
            for i, doc in enumerate(results[:3]):
                content = doc.page_content[:100].replace("\n", " ")
                print(f"    {i+1}. {content}...")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    compare_retrieval_strategies()