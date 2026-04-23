# check_expansion.py


# check_expansion.py — replace with this
from app.ingestion import ingest_all
from app.chunking import recursive_character_chunking, sentence_level_chunking

docs = ingest_all()

policy_docs = [d for d in docs if d.metadata.get("source_type") in ("text", "docx")]
other_docs  = [d for d in docs if d.metadata.get("source_type") not in ("text", "docx")]

policy_chunks = sentence_level_chunking(policy_docs, chunk_size=200, chunk_overlap=20)
other_chunks  = recursive_character_chunking(other_docs, chunk_size=600, chunk_overlap=150)
chunks = policy_chunks + other_chunks

print("\n--- Hotel rate chunks ---")
for i, c in enumerate(chunks):
    if "350" in c.page_content or "250" in c.page_content:
        if "hotel" in c.page_content.lower() or "capped" in c.page_content.lower():
            print(f"\nChunk {i} | chars: {len(c.page_content)}")
            print(c.page_content)
            print("---")

print("\n--- On-call stipend chunks ---")
for i, c in enumerate(chunks):
    if "500" in c.page_content and "on-call" in c.page_content.lower():
        print(f"\nChunk {i} | chars: {len(c.page_content)}")
        print(c.page_content)
        print("---")
    if "200" in c.page_content and "incident" in c.page_content.lower():
        print(f"\nChunk {i} | chars: {len(c.page_content)}")
        print(c.page_content)
        print("---")


# from app.ingestion import ingest_all
# from app.chunking import recursive_character_chunking

# docs = ingest_all()
# chunks = recursive_character_chunking(docs, chunk_size=600, chunk_overlap=150)

# print("\n--- Per diem chunks ---")
# for i, c in enumerate(chunks):
#     if "per diem" in c.page_content.lower():
#         print(f"\nChunk {i} | source: {c.metadata.get('file_name', 'unknown')}")
#         print(f"Chunk {i} | chars: {len(c.page_content)}")
#         print(c.page_content)
#         print("---")


# from app.ingestion import ingest_all
# from app.chunking import recursive_character_chunking, sentence_level_chunking

# docs = ingest_all()

# policy_docs = [d for d in docs if d.metadata.get("source_type") in ("text", "docx")]
# other_docs  = [d for d in docs if d.metadata.get("source_type") not in ("text", "docx")]

# policy_chunks = sentence_level_chunking(policy_docs, chunk_size=100, chunk_overlap=10)
# other_chunks  = recursive_character_chunking(other_docs, chunk_size=600, chunk_overlap=150)
# chunks = policy_chunks + other_chunks

# print(f"\nPolicy chunks: {len(policy_chunks)} | Other: {len(other_chunks)}")

# print("\n--- Per diem chunks ---")
# for i, c in enumerate(chunks):
#     if "per diem" in c.page_content.lower():
#         print(f"\nChunk {i} | chars: {len(c.page_content)}")
#         print(c.page_content)
#         print("---")

# print("\n--- On-call chunks ---")
# for i, c in enumerate(chunks):
#     if "on-call" in c.page_content.lower() and ("500" in c.page_content or "200" in c.page_content):
#         print(f"\nChunk {i} | chars: {len(c.page_content)}")
#         print(c.page_content)
#         print("---")