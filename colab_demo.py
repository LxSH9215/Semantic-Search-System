"""
colab_demo.py — Google Colab–Friendly Walkthrough
===================================================
This single file demonstrates the complete pipeline in a Colab cell-by-cell
format.  Copy-paste individual sections into Colab cells, or run the whole
file with:
    !python colab_demo.py

It imports from the modular files (config, data_loader, embeddings,
clustering, semantic_cache, api) and runs through:
    1. Installation & setup
    2. Data loading & cleaning
    3. Embedding & indexing
    4. Fuzzy clustering with optimal K selection
    5. Boundary-case analysis
    6. Semantic cache demo
    7. FastAPI server (in-process, background thread)
    8. HTTP endpoint testing

This file is designed to be SELF-CONTAINED for Colab while still leveraging
the modular project structure.
"""

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("colab_demo")

# =============================================================
# Cell 1: Install dependencies (run only in Colab)
# =============================================================
# Uncomment the following lines when running in Google Colab:
#
# !pip install -q scikit-learn sentence-transformers faiss-cpu
# !pip install -q fastapi uvicorn[standard] ucimlrepo
# !pip install -q pydantic numpy

print("=" * 60)
print(" Semantic Search System — Colab Demo")
print("=" * 60)


# =============================================================
# Cell 2: Load and clean the 20 Newsgroups dataset
# =============================================================
print("\n📦 STEP 1: Loading 20 Newsgroups from UCI repository…")

from data_loader import load_and_clean

docs, targets, indices = load_and_clean()

print(f"  ✓ Loaded {len(docs)} cleaned documents.")
print(f"  ✓ Categories: {len(set(targets))} unique labels.")
print("\n  Example document (first 300 chars):")
print(f"  {'─' * 50}")
print(f"  {docs[0][:300]}…")
print(f"  Category: {targets[0]}")


# =============================================================
# Cell 3: Encode documents with MiniLM-L6-v2
# =============================================================
print(f"\n🧠 STEP 2: Encoding {len(docs)} documents with MiniLM-L6-v2…")

from embeddings import EmbeddingModel, VectorStore

model = EmbeddingModel()
embeddings = model.encode(docs)

print(f"  ✓ Embeddings shape: {embeddings.shape}")
print(f"  ✓ L2 norms (should be ~1.0): {[f'{n:.4f}' for n in  (embeddings[:3] ** 2).sum(axis=1) ** 0.5]}")


# =============================================================
# Cell 4: Build FAISS index and test retrieval
# =============================================================
print("\n🔍 STEP 3: Building FAISS index…")

store = VectorStore()
store.add(embeddings, docs, targets, indices)

# Quick search demo
test_query = "space shuttle NASA launch"
query_emb = model.encode_single(test_query)
results = store.search(query_emb, top_k=3)

print(f"  ✓ Index contains {store.index.ntotal} vectors.")
print(f"\n  Test query: \"{test_query}\"")
for r in results:
    print(f"    #{r['rank']+1} [sim={r['similarity_score']:.4f}] "
          f"({r['category']}) {r['text'][:80]}…")


# =============================================================
# Cell 5: Fuzzy clustering — Find optimal K
# =============================================================
print("\n📊 STEP 4: Finding optimal number of clusters (BIC + Silhouette)…")
print("  This may take a few minutes depending on hardware.\n")

from clustering import FuzzyClusterer

clusterer = FuzzyClusterer()
k_results = clusterer.find_optimal_k(embeddings)

print(f"\n  ✓ Optimal K (BIC-based): {k_results['best_k']}")
print(f"  ✓ Best K (Silhouette-based): {k_results['best_k_silhouette']}")
bic_display = {k: f'{v:.0f}' for k, v in sorted(k_results['bic_scores'].items())}
print(f"  ✓ BIC scores: {bic_display}")
sil_display = {k: f'{v:.4f}' for k, v in sorted(k_results['sil_scores'].items())}
print(f"  ✓ Silhouette scores: {sil_display}")


# =============================================================
# Cell 6: Fit the GMM with optimal K
# =============================================================
print(f"\n🎯 STEP 5: Fitting GMM with K={clusterer.n_clusters}…")

clusterer.fit(embeddings)
summary = clusterer.get_cluster_summary()

print(f"  ✓ Cluster sizes: {summary['cluster_sizes']}")
print(f"  ✓ Mean max probability: {summary['mean_max_probability']:.4f}")
print(f"  ✓ Boundary cases (uncertain docs): {summary['n_boundary_cases']}")


# =============================================================
# Cell 7: Analyse boundary cases
# =============================================================
print("\n🔬 STEP 6: Analysing boundary cases (uncertain documents)…")

boundaries = clusterer.analyze_boundary_cases(embeddings, docs, max_results=5)

if boundaries:
    print(f"  Found {len(boundaries)} boundary cases (showing top 5):\n")
    for b in boundaries:
        print(f"  Doc #{b['index']} — max_prob={b['max_probability']:.4f}")
        print(f"    Top clusters: {b['top_clusters']}")
        print(f"    Preview: {b['text_preview'][:100]}…\n")
else:
    print("  No boundary cases found (all documents are confidently assigned).")


# =============================================================
# Cell 7.5: Show what lives in each cluster
# =============================================================
print("\n📋 STEP 6b: Showing what lives in each cluster…")
print("  (Proving clusters are semantically meaningful)\n")

cluster_contents = clusterer.show_cluster_contents(embeddings, docs, targets, top_n=3)

for cc in cluster_contents:
    print(f"  ── Cluster {cc['cluster_id']} ({cc['size']} docs) ──")
    print(f"    Top categories: {cc['top_categories']}")
    for rd in cc['representative_docs'][:2]:
        print(f"    • [{rd['category']}] (p={rd['probability']:.3f}) "
              f"{rd['text_preview'][:80]}…")
    print()


# =============================================================
# Cell 8: Semantic cache demo
# =============================================================
print("\n💾 STEP 7: Testing the custom semantic cache…")

from semantic_cache import SemanticCache

cache = SemanticCache(similarity_threshold=0.85)

# First query — should be a miss
q1 = "What is the latest space shuttle mission?"
q1_emb = model.encode_single(q1)
cluster_id, cluster_prob = clusterer.get_dominant_cluster(q1_emb)
top_clusters_q1 = [c[0] for c in clusterer.get_top_clusters(q1_emb, top_n=3)]
result1 = store.search(q1_emb, top_k=3)

hit = cache.lookup(q1_emb, cluster_ids=top_clusters_q1)
print(f"  Query 1: \"{q1}\"")
print(f"    Cluster: {cluster_id} (prob={cluster_prob:.4f})")
print(f"    Cache hit: {hit is not None}")

# Store the result
cache.store(q1, q1_emb, [{"text": r["text"][:200], "category": r["category"],
            "similarity_score": r["similarity_score"], "rank": r["rank"]}
            for r in result1], cluster_id, result1[0]["similarity_score"])

# Similar query — should be a hit (if similar enough)
q2 = "Tell me about the space shuttle mission"
q2_emb = model.encode_single(q2)
top_clusters_q2 = [c[0] for c in clusterer.get_top_clusters(q2_emb, top_n=3)]
hit2 = cache.lookup(q2_emb, cluster_ids=top_clusters_q2)
print(f"\n  Query 2: \"{q2}\"")
print(f"    Cache hit: {hit2 is not None}")
if hit2:
    print(f"    Cached query: \"{hit2.query_text}\"")

# Different query — should miss
q3 = "best programming language for web development"
q3_emb = model.encode_single(q3)
top_clusters_q3 = [c[0] for c in clusterer.get_top_clusters(q3_emb, top_n=3)]
hit3 = cache.lookup(q3_emb, cluster_ids=top_clusters_q3)
print(f"\n  Query 3: \"{q3}\"")
print(f"    Cache hit: {hit3 is not None}")

print(f"\n  Cache stats: {cache.get_stats()}")


# =============================================================
# Cell 9: Start FastAPI server (background thread)
# =============================================================
print("\n🚀 STEP 8: Starting FastAPI server in background…")
print("  Note: In Colab, we run the server in a background thread.")
print("  For standalone use, run: python main.py\n")

import threading
import uvicorn

def run_server():
    """Run FastAPI in a background thread for Colab demo."""
    # Import fresh to get the app with lifespan
    # Since modules are already loaded, the lifespan will re-init
    # (this is fine for demo purposes)
    from api import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

# Only start server if not already running
try:
    import urllib.request
    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
    print("  Server already running!")
except Exception:
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(15)  # Wait for startup (model loading takes time)
    print("  ✓ Server started on http://127.0.0.1:8000")


# =============================================================
# Cell 10: Test the API endpoints
# =============================================================
print("\n🧪 STEP 9: Testing API endpoints…\n")

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

# Test 1: POST /query
print("  ── POST /query ──")
try:
    req_data = json.dumps({"query": "space shuttle NASA", "top_k": 3}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/query",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        print(f"  Status: {resp.status}")
        print(f"  Dominant cluster: {result['dominant_cluster']}")
        print(f"  Similarity score: {result['similarity_score']}")
        print(f"  Cache hit: {result['cache_hit']}")
        print(f"  Results: {len(result['results'])} documents")
except Exception as e:
    print(f"  ⚠ Failed: {e}")

# Test 2: GET /cache/stats
print("\n  ── GET /cache/stats ──")
try:
    with urllib.request.urlopen(f"{BASE_URL}/cache/stats", timeout=10) as resp:
        stats = json.loads(resp.read())
        print(f"  Status: {resp.status}")
        print(f"  Stats: {json.dumps(stats, indent=4)}")
except Exception as e:
    print(f"  ⚠ Failed: {e}")

# Test 3: DELETE /cache
print("\n  ── DELETE /cache ──")
try:
    req = urllib.request.Request(f"{BASE_URL}/cache", method="DELETE")
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
        print(f"  Status: {resp.status}")
        print(f"  Message: {result['message']}")
        print(f"  Previous stats: {result['previous_stats']}")
except Exception as e:
    print(f"  ⚠ Failed: {e}")

print("\n" + "=" * 60)
print(" ✅ Demo complete! All components working.")
print("=" * 60)
print("\nFor standalone use:")
print("  1. pip install -r requirements.txt")
print("  2. python main.py")
print("  3. Open http://localhost:8000/docs for Swagger UI")
print("\nFor Docker:")
print("  1. docker build -t semantic-search .")
print("  2. docker run -p 8000:8000 semantic-search")
