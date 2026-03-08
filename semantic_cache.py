"""
semantic_cache.py — Custom Semantic Cache (No External Libraries)
=================================================================
Design decisions:

Why build from scratch?
  • The task explicitly forbids Redis, Memcached, or caching libraries.
  • A pure-Python implementation lets us deeply integrate with our GMM
    clustering structure, which is the key insight: instead of searching
    ALL cached entries, we only search entries in the query's dominant
    cluster — reducing lookup time from O(N) to O(N/K).

Architecture:
  ┌─────────────────────────────────────────────────┐
  │                  SemanticCache                  │
  │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
  │  │ Cluster 0 │  │ Cluster 1 │  │  …  K−1   │   │
  │  │ [(emb,res)]│ │ [(emb,res)]│ │ [(emb,res)]│  │
  │  └───────────┘  └───────────┘  └───────────┘   │
  └─────────────────────────────────────────────────┘

  On a query:
    1. Compute the query's dominant cluster via GMM.
    2. Search ONLY that cluster's bucket for a cached entry with
       cosine similarity ≥ threshold.
    3. If hit → return cached results (O(bucket_size) instead of O(total)).
    4. If miss → compute fresh results, then store in the bucket.

Tunable similarity threshold:
  • The threshold controls precision vs. recall of cache hits.
  • Higher threshold (e.g., 0.95) = fewer hits, but results are very
    relevant to the exact query.
  • Lower threshold (e.g., 0.80) = more hits, saving compute, but
    the user may get results for a slightly different query.
  • Default: 0.90 — a good balance for semantic search.

Why per-cluster bucketing improves lookup efficiency:
  • As the cache grows, a flat list would require O(N) cosine
    comparisons per query.
  • With K clusters, the average bucket size is N/K, so lookups
    are K× faster.
  • This benefit grows linearly with cache size.

TTL (Time-to-Live) eviction:
  • Optional: if CACHE_TTL_SECONDS > 0, entries older than TTL
    are lazily evicted on the next lookup in that bucket.
  • This prevents stale results if the underlying index changes.
"""

import time
import logging
from typing import List, Dict, Optional, Any

import numpy as np

import config

logger = logging.getLogger(__name__)


class CacheEntry:
    """
    A single cached query with its embedding, results, and timestamp.

    Using a dataclass-style object rather than a raw tuple for clarity.
    """
    __slots__ = ("query_text", "query_embedding", "results", "timestamp",
                 "dominant_cluster", "similarity_score")

    def __init__(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        results: List[Dict],
        dominant_cluster: int,
        similarity_score: float,
    ):
        self.query_text = query_text
        self.query_embedding = query_embedding      # 1-D float32
        self.results = results                       # search results
        self.timestamp = time.time()
        self.dominant_cluster = dominant_cluster
        self.similarity_score = similarity_score


class SemanticCache:
    """
    Cluster-aware semantic cache for query results.

    Uses the GMM clustering structure to partition cached entries into
    per-cluster buckets, dramatically reducing lookup cost as the cache
    grows.
    """

    def __init__(
        self,
        similarity_threshold: float = config.CACHE_SIMILARITY_THRESHOLD,
        max_entries_per_cluster: int = config.CACHE_MAX_ENTRIES_PER_CLUSTER,
        ttl_seconds: float = config.CACHE_TTL_SECONDS,
    ):
        """
        Parameters
        ----------
        similarity_threshold : float
            Minimum cosine similarity for a cache hit (0.0–1.0).
            Higher = stricter matching.
        max_entries_per_cluster : int
            Maximum cached entries per cluster bucket.
            When exceeded, the oldest entry in that bucket is evicted (LRU).
        ttl_seconds : float
            Time-to-live in seconds.  0 = no expiry.
        """
        self.similarity_threshold = similarity_threshold
        self.max_entries_per_cluster = max_entries_per_cluster
        self.ttl_seconds = ttl_seconds

        # Core data structure: cluster_id → list of CacheEntry
        # Using a plain dict for O(1) bucket access
        self._buckets: Dict[int, List[CacheEntry]] = {}

        # Statistics tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0,
            "evictions": 0,
        }

    # ─────────────────────────────────────────────────────────
    # Core operations
    # ─────────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_ids: List[int],
    ) -> Optional[CacheEntry]:
        """
        Check if a semantically similar query exists in the cache.

        Searches the buckets for ALL provided cluster IDs (typically the
        top-N clusters from GMM).  This is critical because semantically
        similar queries can land in different dominant clusters due to subtle
        phrasing differences (e.g., "gun laws" vs "firearm legislation").

        Searching top-3 clusters keeps cost bounded at O(3 × bucket_size)
        while dramatically reducing false misses.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (dim,)
            L2-normalised query embedding.
        cluster_ids : List[int]
            Cluster IDs to search (from GMM top-N clusters).

        Returns
        -------
        CacheEntry or None
            The matching cached entry, or None on cache miss.
        """
        self._stats["total_queries"] += 1

        best_entry = None
        best_sim = -1.0

        for cluster_id in cluster_ids:
            bucket = self._buckets.get(cluster_id, [])
            if not bucket:
                continue

            # Lazy TTL eviction: remove expired entries before searching
            if self.ttl_seconds > 0:
                bucket = self._evict_expired(cluster_id, bucket)

            # Search for best match within this cluster bucket
            for entry in bucket:
                # Cosine similarity between L2-normalised vectors = dot product
                sim = float(np.dot(query_embedding, entry.query_embedding))
                if sim >= self.similarity_threshold and sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        if best_entry is not None:
            self._stats["hits"] += 1
            logger.debug("Cache HIT: sim=%.4f, query='%s'",
                         best_sim, best_entry.query_text[:50])
            return best_entry
        else:
            self._stats["misses"] += 1
            logger.debug("Cache MISS: best_sim=%.4f (threshold=%.4f)",
                         best_sim, self.similarity_threshold)
            return None

    def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        results: List[Dict],
        dominant_cluster: int,
        similarity_score: float,
    ) -> None:
        """
        Store a new query and its results in the appropriate cluster bucket.

        If the bucket exceeds max_entries_per_cluster, the oldest entry
        (lowest timestamp) is evicted — a simple LRU-like policy.
        """
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding,
            results=results,
            dominant_cluster=dominant_cluster,
            similarity_score=similarity_score,
        )

        if dominant_cluster not in self._buckets:
            self._buckets[dominant_cluster] = []

        bucket = self._buckets[dominant_cluster]
        bucket.append(entry)

        # Evict oldest if bucket is full
        if len(bucket) > self.max_entries_per_cluster:
            bucket.sort(key=lambda e: e.timestamp)
            evicted = bucket.pop(0)
            self._stats["evictions"] += 1
            logger.debug("Evicted oldest entry from cluster %d: '%s'",
                         dominant_cluster, evicted.query_text[:50])

    # ─────────────────────────────────────────────────────────
    # TTL eviction
    # ─────────────────────────────────────────────────────────

    def _evict_expired(
        self,
        cluster_id: int,
        bucket: List[CacheEntry],
    ) -> List[CacheEntry]:
        """
        Remove entries older than TTL from a bucket.
        Returns the filtered bucket (also updates self._buckets in place).
        """
        now = time.time()
        before = len(bucket)
        bucket = [e for e in bucket if (now - e.timestamp) < self.ttl_seconds]
        evicted = before - len(bucket)
        if evicted > 0:
            self._stats["evictions"] += evicted
            logger.debug("TTL evicted %d entries from cluster %d.", evicted, cluster_id)
        self._buckets[cluster_id] = bucket
        return bucket

    # ─────────────────────────────────────────────────────────
    # Management / statistics
    # ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        self._buckets.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0,
            "evictions": 0,
        }
        logger.info("Cache cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics including hit rate.

        Returns
        -------
        dict with keys: hits, misses, total_queries, evictions,
                        hit_rate, cache_size, n_buckets
        """
        total = self._stats["total_queries"]
        hit_rate = (self._stats["hits"] / total) if total > 0 else 0.0

        total_entries = sum(len(b) for b in self._buckets.values())

        return {
            **self._stats,
            "hit_rate": round(hit_rate, 4),
            "cache_size": total_entries,
            "n_buckets": len(self._buckets),
            "similarity_threshold": self.similarity_threshold,
        }

    def set_threshold(self, threshold: float) -> None:
        """
        Dynamically adjust the similarity threshold.

        Useful for experimentation without restarting the service.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be in [0.0, 1.0]")
        old = self.similarity_threshold
        self.similarity_threshold = threshold
        logger.info("Cache threshold changed: %.4f → %.4f", old, threshold)


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cache = SemanticCache(similarity_threshold=0.90)

    # Simulate storing and looking up a query
    rng = np.random.RandomState(42)
    fake_emb = rng.randn(config.EMBEDDING_DIM).astype(np.float32)
    fake_emb /= np.linalg.norm(fake_emb)  # L2-normalise

    cache.store(
        query_text="test query",
        query_embedding=fake_emb,
        results=[{"text": "result 1"}],
        dominant_cluster=0,
        similarity_score=0.95,
    )

    # Lookup with the same embedding → should be a hit
    hit = cache.lookup(fake_emb, cluster_ids=[0])
    assert hit is not None, "Expected cache hit"
    print(f"✓ Cache hit: '{hit.query_text}'")

    # Lookup with a random embedding → should miss
    random_emb = rng.randn(config.EMBEDDING_DIM).astype(np.float32)
    random_emb /= np.linalg.norm(random_emb)
    miss = cache.lookup(random_emb, cluster_ids=[0])
    assert miss is None, "Expected cache miss"
    print("✓ Cache miss (as expected)")

    stats = cache.get_stats()
    print(f"✓ Stats: {stats}")
    assert stats["hits"] == 1
    assert stats["misses"] == 1

    cache.clear()
    assert cache.get_stats()["cache_size"] == 0
    print("✓ Cache cleared successfully.")
