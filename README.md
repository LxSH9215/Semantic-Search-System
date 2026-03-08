# Semantic Search System — Trademarkia AI&ML Engineer Task

A modular semantic search system built on the **20 Newsgroups** dataset with fuzzy clustering, a custom semantic cache, and a FastAPI REST API.

## Architecture

```
config.py          ← Central configuration (all tunables in one place)
data_loader.py     ← Load & clean 20 Newsgroups from UCI / sklearn
embeddings.py      ← MiniLM-L6-v2 embeddings + FAISS vector store
clustering.py      ← GMM fuzzy clustering + optimal K + boundary analysis
semantic_cache.py  ← Cluster-aware semantic cache (pure Python, no Redis)
api.py             ← FastAPI service with 3 endpoints
main.py            ← Standalone entry point (python main.py)
colab_demo.py      ← Google Colab walkthrough
```

## Quick Start

### Local Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python main.py
```

The API will be available at **http://localhost:8000**.  
Interactive docs: **http://localhost:8000/docs** (Swagger UI)

### Docker

```bash
# Build and run
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search

# Or use docker-compose
docker-compose up --build
```

## API Endpoints

### `POST /query`

Semantic search with cache integration.

```json
// Request
{ "query": "space shuttle NASA launch", "top_k": 5 }

// Response
{
  "query": "space shuttle NASA launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.8234,
  "dominant_cluster": 3,
  "cluster_probability": 0.87,
  "results": [...]
}
```

### `GET /cache/stats`

Cache hit/miss statistics.

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

Flush the cache entirely and reset all stats.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **MiniLM-L6-v2** embedding model | 384-dim, ~80 MB, strong semantic quality — good speed/quality balance |
| **FAISS IndexFlatIP** | Exact cosine similarity; fast enough for ~20K docs on CPU |
| **GMM** instead of K-Means | Soft assignments — documents get probability distribution across clusters |
| **BIC** for optimal K | Penalises model complexity, guards against overfitting |
| **Cluster-bucketed cache** | Reduces lookup from O(N) to O(N/K) as cache grows |
| **Pure Python cache** | No Redis/Memcached — built from first principles as required |

## Google Colab

Copy `colab_demo.py` into Colab cells, or run:

```bash
!python colab_demo.py
```

## Project Structure

```
├── config.py              # Tuneable constants
├── data_loader.py         # Dataset loading & text cleaning
├── embeddings.py          # Embedding model & FAISS index
├── clustering.py          # GMM fuzzy clustering
├── semantic_cache.py      # Custom semantic cache
├── api.py                 # FastAPI endpoints
├── main.py                # Entry point
├── colab_demo.py          # Colab walkthrough
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container setup
├── docker-compose.yml     # One-command deployment
└── .gitignore
```
