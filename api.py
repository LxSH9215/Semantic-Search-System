"""
api.py — FastAPI Service for Semantic Search
=============================================
This module wires together all components (data loader, embeddings,
clustering, cache) into a REST API with three endpoints.

State management:
  We use module-level singletons initialised in the `lifespan` context
  manager.  This ensures:
    • Components are loaded **once** at startup (not per-request).
    • All requests share the same FAISS index, GMM model, and cache.
    • State is properly cleaned up on shutdown.

  We use FastAPI's modern `lifespan` approach instead of the deprecated
  `on_event("startup")` / `on_event("shutdown")` decorators.

Endpoint design:
  POST /query   — semantic search with cache integration
  GET  /cache/stats — cache hit/miss statistics
  DELETE /cache — clear cache and reset stats
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from data_loader import load_and_clean
from embeddings import EmbeddingModel, VectorStore
from clustering import FuzzyClusterer
from semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Module-level state (populated during lifespan startup)
# ─────────────────────────────────────────────────────────────
embedding_model: Optional[EmbeddingModel] = None
vector_store: Optional[VectorStore] = None
clusterer: Optional[FuzzyClusterer] = None
cache: Optional[SemanticCache] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:
      1. Load and clean the 20 Newsgroups dataset.
      2. Encode all documents with the embedding model.
      3. Build the FAISS index.
      4. Fit GMM clustering on the embeddings.
      5. Initialise the semantic cache.

    On shutdown:
      Log a clean shutdown message.
    """
    global embedding_model, vector_store, clusterer, cache

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info("=== Starting Semantic Search Service ===")

    # Step 1: Load data
    logger.info("Step 1/5: Loading and cleaning dataset…")
    docs, targets, indices = load_and_clean()
    logger.info("  → %d documents ready.", len(docs))

    # Step 2: Encode documents
    logger.info("Step 2/5: Encoding documents…")
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode(docs)
    logger.info("  → Embeddings shape: %s", embeddings.shape)

    # Step 3: Build FAISS index
    logger.info("Step 3/5: Building FAISS index…")
    vector_store = VectorStore()
    vector_store.add(embeddings, docs, targets, indices)
    logger.info("  → Index contains %d vectors.", vector_store.index.ntotal)

    # Step 4: Fit GMM clustering
    logger.info("Step 4/5: Fitting GMM clustering…")
    clusterer = FuzzyClusterer()
    # Use find_optimal_k to determine the best number of clusters
    # This provides evidence (BIC + Silhouette) for the chosen K
    k_results = clusterer.find_optimal_k(embeddings)
    logger.info("  → Optimal K=%d (BIC=%.0f, Sil=%.4f)",
                k_results["best_k"], k_results["best_bic"], k_results["best_sil"])
    clusterer.fit(embeddings)
    logger.info("  → Clustering complete. Summary: %s", clusterer.get_cluster_summary())

    # Step 5: Initialise cache
    logger.info("Step 5/5: Initialising semantic cache…")
    cache = SemanticCache()
    logger.info("  → Cache ready (threshold=%.2f).", cache.similarity_threshold)

    logger.info("=== Service ready! ===")

    yield  # ← Application runs here

    logger.info("=== Shutting down Semantic Search Service ===")


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search Service",
    description=(
        "Semantic search over the 20 Newsgroups dataset with fuzzy clustering "
        "and a custom semantic cache. Built for the Trademarkia AI&ML Engineer Task."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /query endpoint.
    
    The brief specifies: { "query": "<natural language query>" }
    """
    query: str = Field(..., description="The natural language search query.")
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        ge=1, le=50,
        description="Number of results to return.",
    )


class SearchResult(BaseModel):
    """A single search result."""
    text: str
    category: str
    similarity_score: float
    rank: int


class QueryResponse(BaseModel):
    """Response body for the /query endpoint.
    
    Matches the brief's expected response format:
      query, cache_hit, matched_query, similarity_score, result, dominant_cluster
    """
    query: str
    cache_hit: bool
    matched_query: Optional[str] = Field(
        None,
        description="The cached query that was matched (None on cache miss).",
    )
    similarity_score: float
    dominant_cluster: int
    cluster_probability: float
    results: List[SearchResult]


class CacheStatsResponse(BaseModel):
    """Response body for the /cache/stats endpoint.
    
    Field names match the brief's expected response:
      total_entries, hit_count, miss_count, hit_rate
    Plus additional useful fields.
    """
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    total_queries: int
    evictions: int
    n_buckets: int
    similarity_threshold: float


class CacheClearResponse(BaseModel):
    """Response body for the DELETE /cache endpoint."""
    message: str
    previous_stats: dict


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Perform a semantic search query.

    Flow:
      1. Encode the query text.
      2. Determine the dominant cluster via GMM.
      3. Check the semantic cache for a hit.
      4. On miss: search FAISS, store result in cache.
      5. Return results with cluster info and cache status.
    """
    # Step 1: Encode the query
    query_emb = embedding_model.encode_single(request.query)

    # Step 2: Determine dominant cluster
    dominant_cluster, cluster_prob = clusterer.get_dominant_cluster(query_emb)

    # Step 3: Check cache
    cached = cache.lookup(query_emb, dominant_cluster)

    if cached is not None:
        # Cache HIT — return cached results with the matched_query
        # matched_query shows the original cached query that was semantically
        # matched, so the user can see what their query was compared against.
        return QueryResponse(
            query=request.query,
            cache_hit=True,
            matched_query=cached.query_text,
            similarity_score=cached.similarity_score,
            dominant_cluster=cached.dominant_cluster,
            cluster_probability=cluster_prob,
            results=[SearchResult(**r) for r in cached.results],
        )

    # Step 4: Cache MISS — search FAISS
    raw_results = vector_store.search(query_emb, top_k=request.top_k)

    # Format results for response and caching
    results = []
    for r in raw_results:
        results.append({
            "text": r["text"][:500],  # Truncate for response size
            "category": r["category"],
            "similarity_score": round(r["similarity_score"], 4),
            "rank": r["rank"],
        })

    # Top similarity score (from the best FAISS match)
    top_sim = results[0]["similarity_score"] if results else 0.0

    # Step 5: Store in cache
    cache.store(
        query_text=request.query,
        query_embedding=query_emb,
        results=results,
        dominant_cluster=dominant_cluster,
        similarity_score=top_sim,
    )

    return QueryResponse(
        query=request.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=top_sim,
        dominant_cluster=dominant_cluster,
        cluster_probability=cluster_prob,
        results=[SearchResult(**r) for r in results],
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """
    Return cache statistics: hits, misses, hit rate, size.

    This endpoint is useful for monitoring cache effectiveness and
    tuning the similarity threshold.
    """
    stats = cache.get_stats()
    # Map internal stat names to the brief's expected field names
    return CacheStatsResponse(
        total_entries=stats["cache_size"],
        hit_count=stats["hits"],
        miss_count=stats["misses"],
        hit_rate=stats["hit_rate"],
        total_queries=stats["total_queries"],
        evictions=stats["evictions"],
        n_buckets=stats["n_buckets"],
        similarity_threshold=stats["similarity_threshold"],
    )


@app.delete("/cache", response_model=CacheClearResponse)
async def clear_cache():
    """
    Clear the semantic cache and reset all statistics.

    This is a destructive operation — all cached queries are lost.
    Use this when you change the similarity threshold or want a fresh
    start for benchmarking.
    """
    previous = cache.get_stats()
    cache.clear()
    return CacheClearResponse(
        message="Cache cleared successfully.",
        previous_stats=previous,
    )


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "index_size": vector_store.index.ntotal if vector_store else 0,
        "n_clusters": clusterer.n_clusters if clusterer else 0,
        "cache_size": cache.get_stats()["cache_size"] if cache else 0,
    }
