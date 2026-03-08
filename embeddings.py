"""
embeddings.py — Embedding Model + FAISS Vector Store
=====================================================
Design decisions:

Model choice — sentence-transformers/all-MiniLM-L6-v2:
  • 384-dimensional output → compact vectors, fast FAISS searches.
  • ~22 M parameters (~80 MB) → loads in <5 s even on CPU.
  • Trained on >1 B sentence pairs with a contrastive objective,
    so it produces high-quality semantic embeddings out of the box.
  • Well-supported by the sentence-transformers library (active maintenance).

Index choice — FAISS IndexFlatIP (inner product on L2-normalised vectors):
  • This gives **exact** cosine similarity search.
  • For ~20 K documents, brute-force search takes <10 ms on CPU,
    so there is no need for approximate indices (IVF, HNSW).
  • If the corpus grew to millions, we would swap in IndexIVFFlat
    with nprobe tuning — but that is premature here.

Why L2-normalise?
  • Cosine similarity = dot product when vectors are unit-length.
  • FAISS's IndexFlatIP is heavily SIMD-optimised for dot products,
    making it faster than computing cosine from scratch.
"""

import logging
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer that handles encoding
    and L2-normalisation in one place.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = config.EMBEDDING_DIM

    def encode(
        self,
        texts: List[str],
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised embeddings.

        Parameters
        ----------
        texts : List[str]
            Raw text strings to embed.
        batch_size : int
            Number of texts per forward pass (tune for GPU memory).
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        embeddings : np.ndarray of shape (len(texts), EMBEDDING_DIM)
            L2-normalised float32 embeddings.
        """
        logger.info("Encoding %d texts (batch_size=%d)…", len(texts), batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # ← built-in L2 normalisation
        )
        embeddings = embeddings.astype(np.float32)
        logger.info("Encoding complete. Shape: %s", embeddings.shape)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience: encode one text and return a 1-D vector."""
        return self.encode([text], show_progress=False)[0]


class VectorStore:
    """
    FAISS-backed vector store for fast cosine-similarity retrieval.

    The store keeps a parallel list of document metadata (text, category,
    original index) so that search results are immediately useful.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        # IndexFlatIP = brute-force inner-product search.
        # Combined with L2-normalised vectors this gives exact cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.documents: List[dict] = []   # parallel metadata store
        self.dim = dim

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        targets: List[str],
        original_indices: List[int],
    ) -> None:
        """
        Add document embeddings and their metadata to the index.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, dim)
        documents : List[str]   — cleaned document texts
        targets : List[str]     — category labels
        original_indices : List[int] — indices in the raw dataset
        """
        assert embeddings.shape[0] == len(documents), "Count mismatch"
        assert embeddings.shape[1] == self.dim, f"Expected dim {self.dim}"

        self.index.add(embeddings)
        for doc, tgt, oidx in zip(documents, targets, original_indices):
            self.documents.append({
                "text": doc,
                "category": tgt,
                "original_index": oidx,
            })
        logger.info("Index now contains %d vectors.", self.index.ntotal)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> List[dict]:
        """
        Find the top-K most similar documents to the query embedding.

        Returns a list of dicts, each containing:
          - text, category, original_index  (from the stored metadata)
          - similarity_score  (cosine similarity, in [−1, 1])
          - rank              (0-based)
        """
        # FAISS expects a 2-D query matrix
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots
            result = dict(self.documents[idx])
            result["similarity_score"] = float(score)
            result["rank"] = rank
            results.append(result)

        return results

    def save(self, path: str = config.FAISS_INDEX_PATH) -> None:
        """Persist the FAISS index to disk."""
        faiss.write_index(self.index, path)
        logger.info("FAISS index saved to %s", path)

    def load(self, path: str = config.FAISS_INDEX_PATH) -> None:
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)
        logger.info("FAISS index loaded from %s (%d vectors).",
                     path, self.index.ntotal)


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = EmbeddingModel()
    sample_texts = [
        "The space shuttle launched successfully.",
        "Machine learning models require large datasets.",
        "The stock market crashed today.",
    ]
    embs = model.encode(sample_texts, show_progress=False)
    print(f"✓ Encoded {len(sample_texts)} texts → shape {embs.shape}")

    store = VectorStore()
    store.add(embs, sample_texts, ["sci", "comp", "misc"], [0, 1, 2])

    query_emb = model.encode_single("NASA rocket launch")
    results = store.search(query_emb, top_k=2)
    print(f"✓ Top result for 'NASA rocket launch': "
          f"score={results[0]['similarity_score']:.3f}, "
          f"text='{results[0]['text'][:60]}'")
