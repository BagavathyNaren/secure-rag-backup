# app/data_indexing.py

import time
from tracemalloc import start
import numpy as np
from typing import List
from langchain_core.documents import Document
from app.config import *
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking
from langchain_openai import OpenAIEmbeddings
import faiss


# 1. GENERATE EMBEDDINGS FOR ALL CHUNKS

def prepare_data():
    chunks = recursive_character_chunking(ingest_all())
    
    model = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [chunk.page_content for chunk in chunks]
    vectors = model.embed_documents(texts)
    vectors_np = np.array(vectors).astype("float32")
    
    print(f"  Loaded {len(chunks)} chunks → {vectors_np.shape[0]} embedding vectors")
    
    return chunks, vectors_np, model

if __name__ == "__main__":
    chunks, vectors, model = prepare_data()
    print(f"Vector shape: {vectors.shape}")  # Should be (214, 1536)

# 2. FAISS INDEX TYPES COMPARISON

def compare_faiss_indexes(vectors: np.ndarray, chunks: List[Document]):

    dimension = vectors.shape[1]
    n_vectors = vectors.shape[0]

    print(f"\n  Dataset: {n_vectors} vectors, {dimension} dimensions")
    print(f"  Raw data size: {vectors.nbytes / 1024:.1f} KB")

    query_vectors = vectors[:5]
    results = {}

    # --- A) FLAT INDEX (Ground Truth) ---
    start = time.time()
    flat_index = faiss.IndexFlatL2(dimension)
    flat_index.add(vectors)
    build_time = time.time() - start

    start = time.time()
    for _ in range(5):
        D, I = flat_index.search(query_vectors, 3)
    query_time = (time.time() - start) / 5

    results["Flat"] = {
        "build_time": build_time,
        "query_time": query_time,
        "indices": I,
        "recall": 1.0,
        "memory_kb": vectors.nbytes / 1024,
    }

    print(f"  Flat index built. Build: {build_time:.4f}s | Query: {query_time:.5f}s")

    # --- B) HNSW INDEX ---
    start = time.time()
    hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.hnsw.efSearch = 64
    hnsw_index.add(vectors)
    build_time = time.time() - start

    start = time.time()
    for _ in range(5):
        D, I = hnsw_index.search(query_vectors, 3)
    query_time = (time.time() - start) / 5

    # Calculate recall vs flat (how many results match ground truth)
    flat_indices = results["Flat"]["indices"]
    matches = sum(len(set(I[i]) & set(flat_indices[i])) for i in range(len(query_vectors)))
    recall = matches / (len(query_vectors) * 3)

    results["HNSW"] = {
    "build_time": build_time,
    "query_time": query_time,
    "indices": I,
    "recall": recall,
    "memory_kb": vectors.nbytes / 1024 * 1.2,  # approx: vectors + graph overhead
      }

    print(f"  HNSW index built.  Build: {build_time:.4f}s | Query: {query_time:.5f}s | Recall: {recall:.0%}")

    
     # --- C) IVF INDEX ---

    nlist = 10  # number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    start = time.time()
    ivf_index.train(vectors)  # IVF MUST be trained first
    ivf_index.add(vectors)
    build_time = time.time() - start

    ivf_index.nprobe = 3  # search 3 clusters at query time

    start = time.time()
    for _ in range(5):
        D, I = ivf_index.search(query_vectors, 3)
    query_time = (time.time() - start) / 5

    flat_indices = results["Flat"]["indices"]
    matches = sum(len(set(I[i]) & set(flat_indices[i])) for i in range(len(query_vectors)))
    recall = matches / (len(query_vectors) * 3)

    results["IVF"] = {
    "build_time": build_time,
    "query_time": query_time,
    "indices": I,
    "recall": recall,
    "memory_kb": vectors.nbytes / 1024,
    }

    print(f"  IVF index built.   Build: {build_time:.4f}s | Query: {query_time:.5f}s | Recall: {recall:.0%}")

    # --- D) IVF-PQ INDEX ---
    nlist = 10
    m = 48  # 1536 / 48 = 32 sub-vectors (must divide evenly)
    quantizer = faiss.IndexFlatL2(dimension)

    # 4 bits per sub-vector
    ivfpq_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 4)  

        # 8 bits per sub-vector
    # ivfpq_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8) 

    start = time.time()
    ivfpq_index.train(vectors)
    ivfpq_index.add(vectors)
    build_time = time.time() - start

    ivfpq_index.nprobe = 3

    start = time.time()
    for _ in range(5):
        D, I = ivfpq_index.search(query_vectors, 3)
    query_time = (time.time() - start) / 5

    flat_indices = results["Flat"]["indices"]
    matches = sum(len(set(I[i]) & set(flat_indices[i])) for i in range(len(query_vectors)))
    recall = matches / (len(query_vectors) * 3)

    results["IVF-PQ"] = {
        "build_time": build_time,
        "query_time": query_time,
        "indices": I,
        "recall": recall,
        "memory_kb": (n_vectors * m) / 1024,  # compressed size
    }

    print(f"  IVF-PQ index built. Build: {build_time:.4f}s | Query: {query_time:.5f}s | Recall: {recall:.0%}")

    # --- PRINT COMPARISON TABLE ---
    print(f"\n  {'Index':<10} {'Build':>10} {'Query':>10} {'Recall':>10} {'Memory':>10}")
    print(f"  {'-'*50}")
    for name, r in results.items():
        print(f"  {name:<10} {r['build_time']:>9.4f}s {r['query_time']:>9.5f}s {r['recall']:>9.0%} {r['memory_kb']:>8.0f} KB")
    print(f"\n  Note: Differences become dramatic at 1M+ vectors.")

# ============================================================
# 3. QDRANT INDEX CONFIGURATION
# ============================================================
def demonstrate_qdrant_indexing(chunks: List[Document]):
    """
    Shows how to configure HNSW parameters in Qdrant and how 
    metadata indexing enables filtered search.
    
    Steps:
    1. Create a Qdrant collection with custom HNSW config
    2. Create metadata (payload) indexes for filtered search
    3. Demonstrate filtered retrieval
    
    Use:
        from qdrant_client import QdrantClient, models
        
        client = QdrantClient(host="localhost", port=6333)
        
        # Create collection with custom HNSW config
        client.create_collection(
            collection_name="indexed_docs",
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=16,                    # connections per node
                    ef_construct=100,        # build quality
                ),
            ),
        )
        
        # Create payload (metadata) indexes for fast filtering
        client.create_payload_index(
            collection_name="indexed_docs",
            field_name="source_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name="indexed_docs",
            field_name="file_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    
    Then demonstrate filtered search:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client.models import FieldCondition, MatchValue, Filter
        
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="indexed_docs",
            embedding=OpenAIEmbeddings(),
        )
        
        # Search ONLY in text documents
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": Filter(
                    must=[
                        FieldCondition(
                            key="source_type",
                            match=MatchValue(value="text"),
                        )
                    ]
                ),
            }
        )
    """
    # YOUR CODE HERE
    from qdrant_client import QdrantClient, models
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client.models import FieldCondition, MatchValue, Filter

    client = QdrantClient(host="localhost", port=6333)

    # Delete collection if it already exists (clean slate)
    if client.collection_exists("indexed_docs"):
        client.delete_collection("indexed_docs")

    # Create collection with custom HNSW config
    client.create_collection(
        collection_name="indexed_docs",
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        ),
    )
    print("  Created collection with HNSW config (m=16, ef_construct=100)")

    # Create payload indexes for fast filtering
    client.create_payload_index(
        collection_name="indexed_docs",
        field_name="metadata.source_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name="indexed_docs",
        field_name="metadata.file_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    print("  Created payload indexes: source_type, file_name")

    # Upload chunks
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="indexed_docs",
        url="http://localhost:6333",
    )
    print(f"  Uploaded {len(chunks)} vectors with metadata")

    # Demonstrate filtered search
    query = "password policy"
    print(f"\n  Query: '{query}'")

    # Unfiltered
    unfiltered = vectorstore.similarity_search(query, k=3)
    print("\n  Unfiltered (all sources):")
    for doc in unfiltered:
        print(f"    [{doc.metadata.get('source_type')}] {doc.page_content[:60]}...")

    # Filtered to text only
    filtered = vectorstore.similarity_search(
        query,
        k=3,
        filter=Filter(must=[FieldCondition(key="metadata.source_type", match=MatchValue(value="text"))])
    )
    print("\n  Filtered (source_type='text' only):")
    for doc in filtered:
        print(f"    [{doc.metadata.get('source_type')}] {doc.page_content[:60]}...")

    return vectorstore

# ============================================================
# 4. METADATA INDEXING BEST PRACTICES
# ============================================================
def demonstrate_metadata_filtering(vectorstore_with_metadata):
    """
    Shows how metadata indexes enable precise retrieval.
    
    Test queries with filters:
    1. "password policy" filtered to source_type="text" only
    2. "server configuration" filtered to file_name containing "infrastructure"
    3. "travel expenses" filtered to source_type="csv" 
       (should return sales data, not policy text)
    
    For each, compare:
    - Unfiltered results (may include noise)
    - Filtered results (precise, relevant)
    """
    # YOUR CODE HERE
    from qdrant_client.models import FieldCondition, MatchValue, Filter

    queries = [
        {
            "query": "password policy",
            "filter": Filter(must=[FieldCondition(key="metadata.source_type", match=MatchValue(value="text"))]),
            "label": "source_type='text'",
        },
        {
            "query": "server configuration",
            "filter": Filter(must=[FieldCondition(key="metadata.file_name", match=MatchValue(value="infrastructure_inventory.xlsx"))]),
            "label": "file_name='infrastructure_inventory.xlsx'",
        },
        {
            "query": "travel expenses",
            "filter": Filter(must=[FieldCondition(key="metadata.source_type", match=MatchValue(value="csv"))]),
            "label": "source_type='csv'",
        },
    ]

    for q in queries:
        print(f"\n  Query: '{q['query']}' | Filter: {q['label']}")

        unfiltered = vectorstore_with_metadata.similarity_search(q["query"], k=3)
        print("  Unfiltered:")
        for doc in unfiltered:
            print(f"    [{doc.metadata.get('source_type')}] {doc.page_content[:60]}...")

        filtered = vectorstore_with_metadata.similarity_search(q["query"], k=3, filter=q["filter"])
        print("  Filtered:")
        for doc in filtered:
            print(f"    [{doc.metadata.get('source_type')}] {doc.page_content[:60]}...")



# ============================================================
# 5. FULL INDEXING COMPARISON
# ============================================================
def run_indexing_comparison():
    """
    Complete indexing comparison pipeline.
    """
    
    print("=" * 70)
    print("DATA INDEXING DEEP DIVE")
    print("=" * 70)
    
    # Step 1: Prepare data
    print("\n📦 Preparing data...")
    chunks, vectors, embedding_model = prepare_data()
    
    # Step 2: Compare FAISS index types
    print("\n" + "=" * 70)
    print("FAISS INDEX TYPE COMPARISON")
    print("=" * 70)
    compare_faiss_indexes(vectors, chunks)
    
    # Step 3: Qdrant indexing with metadata
    vectorstore = demonstrate_qdrant_indexing(chunks)  # ← must return vectorstore now

    print("\n" + "=" * 70)
    print("METADATA FILTERING BEST PRACTICES")
    print("=" * 70)
    demonstrate_metadata_filtering(vectorstore)
    
    print("\n" + "=" * 70)
    print("INDEXING COMPARISON COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_indexing_comparison()