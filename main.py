"""
main.py — Standalone Entry Point for the Semantic Search Service
================================================================
Run this file to start the FastAPI server:

    python main.py

The server will:
  1. Load the 20 Newsgroups dataset from the UCI repository.
  2. Encode all documents using MiniLM-L6-v2.
  3. Build a FAISS index for similarity search.
  4. Fit a GMM for fuzzy clustering.
  5. Start the API on http://0.0.0.0:8000

Interactive API docs:  http://localhost:8000/docs  (Swagger UI)
"""

import uvicorn
import config


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,          # Disable in production (large model reload is slow)
        log_level="info",
    )
