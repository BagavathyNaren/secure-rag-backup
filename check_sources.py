import sys, os
sys.path.insert(0, '.')
from app.ingestion import ingest_all

docs = ingest_all()

keywords = ['password', 'stipend', 'hotel', 'on-call', 'rotation']
allowed = ['company_policy.txt', 'employee_handbook.pdf', 'engineering_standards.docx']

for doc in docs:
    source = doc.metadata.get('source', '')
    if any(src in source for src in allowed):
        for kw in keywords:
            if kw.lower() in doc.page_content.lower():
                print(f'--- [{source}] ---')
                print(doc.page_content[:300])
                print()
                break