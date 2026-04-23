# app/chunking.py

from typing import List
from langchain_core.documents import Document
from app.ingestion import ingest_all
from app.config import *


def sentence_level_chunking(
    docs: List[Document],
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> List[Document]:
    """
    Splits documents into individual sentences using nltk.
    Each sentence becomes its own chunk regardless of character count.
    This is the only reliable way to separate co-located facts like
    '$75 domestic' and '$100 international' that live in the same paragraph.
    
    Args:
        docs: Policy documents (text, docx source types)
        chunk_size: Unused — kept for API compatibility
        chunk_overlap: Unused — kept for API compatibility
        
    Returns:
        One chunk per sentence
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    result = []
    for doc in docs:
        sentences = nltk.sent_tokenize(doc.page_content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # skip headers, single words, noise
                continue
            result.append(Document(
                page_content=sentence,
                metadata=doc.metadata
            ))
    return result


# ============================================================
# STRATEGY 1: RECURSIVE CHARACTER SPLITTING
# ============================================================
def recursive_character_chunking(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 200,
) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)


# ============================================================
# STRATEGY 2: SEMANTIC CHUNKING
# ============================================================
def semantic_chunking(
    docs: List[Document],
) -> List[Document]:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    chunker = SemanticChunker(
        OpenAIEmbeddings(),
        breakpoint_threshold_type="percentile"
    )
    return chunker.split_documents(docs)


# ============================================================
# STRATEGY 3: PARENT-CHILD CHUNKING
# ============================================================
def parent_child_chunking(
    docs: List[Document],
    parent_chunk_size: int = 1500,
    child_chunk_size: int = 300,
) -> dict:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=200,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=50,
    )

    parent_chunks = parent_splitter.split_documents(docs)

    child_chunks = []
    for parent_id, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
        child_chunks.extend(children)

    return {
        "parent_chunks": parent_chunks,
        "child_chunks": child_chunks,
    }


# ============================================================
# COMPARISON REPORT
# ============================================================
def compare_strategies(docs: List[Document]):
    """
    Runs all three strategies and prints a comparison.
    This function is DONE — do not modify.
    """

    # Filter: only long-text docs for semantic chunking
    long_docs = [
        d for d in docs
        if d.metadata.get("source_type") in ("text", "pdf", "docx", "web")
    ]

    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 70)

    # Strategy 1: Recursive
    recursive_chunks = recursive_character_chunking(docs)
    print(f"\n1. Recursive Character Splitting:")
    print(f"   Input docs:    {len(docs)}")
    print(f"   Output chunks: {len(recursive_chunks)}")
    sizes = [len(c.page_content) for c in recursive_chunks]
    print(f"   Avg chunk size: {sum(sizes)//len(sizes)} chars")
    print(f"   Min chunk size: {min(sizes)} chars")
    print(f"   Max chunk size: {max(sizes)} chars")
    print(f"   Sample chunk:  '{recursive_chunks[0].page_content[:80]}...'")

    # Strategy 2: Semantic
    semantic_chunks = semantic_chunking(long_docs)
    print(f"\n2. Semantic Chunking (long docs only):")
    print(f"   Input docs:    {len(long_docs)}")
    print(f"   Output chunks: {len(semantic_chunks)}")
    sizes = [len(c.page_content) for c in semantic_chunks]
    print(f"   Avg chunk size: {sum(sizes)//len(sizes)} chars")
    print(f"   Min chunk size: {min(sizes)} chars")
    print(f"   Max chunk size: {max(sizes)} chars")
    print(f"   Sample chunk:  '{semantic_chunks[0].page_content[:80]}...'")

    # Strategy 3: Parent-Child
    pc_result = parent_child_chunking(long_docs)
    parents = pc_result["parent_chunks"]
    children = pc_result["child_chunks"]
    print(f"\n3. Parent-Child Chunking (long docs only):")
    print(f"   Input docs:    {len(long_docs)}")
    print(f"   Parent chunks: {len(parents)}")
    print(f"   Child chunks:  {len(children)}")
    p_sizes = [len(c.page_content) for c in parents]
    c_sizes = [len(c.page_content) for c in children]
    print(f"   Avg parent size: {sum(p_sizes)//len(p_sizes)} chars")
    print(f"   Avg child size:  {sum(c_sizes)//len(c_sizes)} chars")

    # Show parent-child relationship
    sample_child = children[0]
    parent_id = sample_child.metadata.get("parent_id", 0)
    print(f"\n   Parent-Child Example:")
    print(f"   Child:  '{sample_child.page_content[:80]}...'")
    print(f"   Parent: '{parents[parent_id].page_content[:80]}...'")

    print("\n" + "=" * 70)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    docs = ingest_all()
    compare_strategies(docs)