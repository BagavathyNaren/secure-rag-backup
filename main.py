# main.py

from db.connection import init_db
from db.seed import seed_users
import json
import logging

logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup():
    
    # 1. Create tables
    await init_db()
    logger.info(json.dumps({
        "trace_id": "startup",
        "event":    "DB_INIT",
        "status":   "tables_ready"
    }))
    
    # 2. Seed mock users (skips if already exist)
    await seed_users()
    logger.info(json.dumps({
        "trace_id": "startup", 
        "event":    "DB_SEED",
        "status":   "complete"
    }))
    
    # 3. Your existing cache + FAISS init
    # ... rest of startup