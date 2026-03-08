"""
config.py — Central Configuration for the Semantic Search System
================================================================
All tuneable constants live here so every module reads from one source of truth.
This avoids "magic numbers" scattered across files and makes experimentation easy.
"""

# ─────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────
# UCI Machine Learning Repository dataset ID for Twenty Newsgroups
# Reference: https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
UCI_DATASET_ID = 113

# Minimum document length (in characters) after cleaning.
# Very short docs (signatures, one-liners) add noise without semantic value.
MIN_DOC_LENGTH = 50

# ─────────────────────────────────────────────────────────────
# 2. Embedding Model
# ─────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is a good balance between quality and speed:
#   • 384-dimensional output  (compact → fast FAISS search)
#   • ~80 MB model size       (fits in free-tier Colab RAM)
#   • Trained on 1B+ sentence pairs → strong semantic understanding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Batch size for encoding documents.  Larger = faster on GPU, but more VRAM.
EMBEDDING_BATCH_SIZE = 256

# ─────────────────────────────────────────────────────────────
# 3. FAISS Index
# ─────────────────────────────────────────────────────────────
# We use IndexFlatIP (inner-product on L2-normalised vectors) which gives
# exact cosine similarity.  For ~20 K docs this is fast enough; for millions
# you would switch to an IVF or HNSW index.
FAISS_INDEX_PATH = "faiss_index.bin"

# Default number of results returned per query
DEFAULT_TOP_K = 5

# ─────────────────────────────────────────────────────────────
# 4. Fuzzy Clustering (Gaussian Mixture Model)
# ─────────────────────────────────────────────────────────────
# Range of cluster counts to evaluate when selecting optimal K.
# 20 Newsgroups has 20 ground-truth categories, but soft clustering may
# find a different optimal K due to topic overlap (e.g., sci.space ≈ sci.med).
CLUSTER_K_MIN = 5
CLUSTER_K_MAX = 30
CLUSTER_K_STEP = 1

# GMM covariance type: "full" captures correlations between embedding dims
# but is expensive; "diag" is a good trade-off for high-dimensional data.
GMM_COVARIANCE_TYPE = "diag"

# Random seed for reproducibility
RANDOM_SEED = 42

# Threshold for boundary-case analysis:
# Documents whose max cluster probability < this value are "uncertain".
BOUNDARY_UNCERTAINTY_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────
# 5. Semantic Cache
# ─────────────────────────────────────────────────────────────
# Cosine-similarity threshold for cache hits.
# Higher = stricter matching (fewer hits but more precise).
# 0.90 is a sensible default — queries must be very similar to get a hit.
CACHE_SIMILARITY_THRESHOLD = 0.90

# Maximum entries per cluster bucket.  Keeps memory bounded.
CACHE_MAX_ENTRIES_PER_CLUSTER = 100

# Time-to-live in seconds for cache entries (0 = no expiry).
CACHE_TTL_SECONDS = 0

# ─────────────────────────────────────────────────────────────
# 6. FastAPI
# ─────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
