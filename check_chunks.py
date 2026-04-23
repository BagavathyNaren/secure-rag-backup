
# paste into a quick script or terminal

from app.ingestion import load_docx_files
from app.chunking import recursive_character_chunking

docs = load_docx_files("data/")
chunks = recursive_character_chunking(docs, chunk_size=600, chunk_overlap=150)

for i, c in enumerate(chunks):
    if "200" in c.page_content and "incident" in c.page_content.lower():
        print(f"Chunk {i}:")
        print(c.page_content)
        print("---")

# from app.ingestion import load_docx_files

# docs = load_docx_files("data/")
# for doc in docs:
#     if doc.metadata.get("file_name") == "engineering_standards.docx":
#         if "500" in doc.page_content or "200" in doc.page_content:
#             print(doc.page_content[:300])
#             print("---")



# # check_chunks.py — replace with this
# from app.ingestion import ingest_all
# from app.chunking import recursive_character_chunking

# docs = ingest_all()
# chunks = recursive_character_chunking(docs, chunk_size=600, chunk_overlap=150)

# print("\n--- Searching raw docs for '$200' ---")
# for doc in docs:
#     if "200" in doc.page_content:
#         print(f"\nSource: {doc.metadata.get('source', 'unknown')}")
#         print(doc.page_content[:400])
#         print("...")

# print("\n--- Searching chunks for '$200' ---")
# for i, c in enumerate(chunks):
#     if "200" in c.page_content:
#         print(f"\nChunk {i} | source: {c.metadata.get('source', 'unknown')}")
#         print(c.page_content)
#         print("---")



# import sys
# sys.path.insert(0, '.')
# from app.ingestion import ingest_all
# from app.chunking import recursive_character_chunking

# docs = ingest_all()
# chunks = recursive_character_chunking(docs)

# for c in chunks:
#     if 'stipend' in c.page_content.lower() or 'on-call' in c.page_content.lower():
#         print(f'--- [{c.metadata.get("source")}] ---')
#         print(c.page_content)
#         print()


# # Quick diagnostic — run this once
# from app.ingestion import ingest_all
# from app.chunking import recursive_character_chunking

# docs = ingest_all()
# chunks = recursive_character_chunking(docs, chunk_size=600, chunk_overlap=150)

# for i, c in enumerate(chunks):
#     if "200" in c.page_content and "incident" in c.page_content.lower():
#         print(f"Chunk {i}: {c.page_content[:200]}")