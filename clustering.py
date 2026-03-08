"""
clustering.py — Fuzzy (Soft) Clustering with Gaussian Mixture Models
=====================================================================
Design decisions:

Why GMM instead of K-Means?
  • K-Means assigns each document to **exactly one** cluster (hard assignment).
  • GMM assigns a **probability distribution** over clusters (soft assignment).
  • This is crucial for the 20 Newsgroups domain: many posts naturally span
    multiple topics (e.g., a post about "encryption law" belongs to both
    sci.crypt and talk.politics).  Soft assignments capture this nuance.

Why "diag" covariance?
  • "full" covariance matrices have O(d²) parameters per component.
    With d=384 dimensions and K~20 clusters, that is ~2.9 M params per
    component — likely to overfit on ~18 K documents.
  • "diag" assumes feature dimensions are conditionally independent given
    the cluster, which is a reasonable assumption for dense embeddings that
    already decorrelate features during training.

Optimal K selection:
  • We use **BIC (Bayesian Information Criterion)** as the primary metric.
    BIC penalises model complexity (number of parameters), so it naturally
    guards against overfitting — unlike raw log-likelihood which always
    improves with more clusters.
  • We also compute **Silhouette scores** as a secondary geometric metric
    that measures how well-separated clusters are in embedding space.
  • We report both so the user can make an informed decision.

Boundary-case analysis:
  • Documents whose maximum cluster probability < BOUNDARY_UNCERTAINTY_THRESHOLD
    are flagged as "uncertain".  These are the most interesting cases for
    human review — they often sit at the intersection of topics.
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import config

logger = logging.getLogger(__name__)


class FuzzyClusterer:
    """
    Gaussian Mixture Model–based fuzzy clustering for document embeddings.

    After fitting, every document has a *distribution* over K clusters
    rather than a single hard assignment.
    """

    def __init__(self, n_clusters: Optional[int] = None):
        """
        Parameters
        ----------
        n_clusters : int or None
            If None, the optimal number is determined automatically via
            `find_optimal_k()`.  If given, the GMM is initialised directly.
        """
        self.n_clusters = n_clusters
        self.gmm: Optional[GaussianMixture] = None
        self.cluster_probabilities: Optional[np.ndarray] = None  # (N, K)
        self.hard_labels: Optional[np.ndarray] = None            # (N,)

    # ─────────────────────────────────────────────────────────
    # Optimal K selection
    # ─────────────────────────────────────────────────────────

    def find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_min: int = config.CLUSTER_K_MIN,
        k_max: int = config.CLUSTER_K_MAX,
        k_step: int = config.CLUSTER_K_STEP,
    ) -> Dict:
        """
        Sweep over K values and select the best using BIC + Silhouette.

        Returns a dict with keys:
          - best_k       : int
          - bic_scores   : dict {k: bic}
          - sil_scores   : dict {k: silhouette}
          - best_bic     : float
          - best_sil     : float
        """
        logger.info("Searching for optimal K in [%d, %d]…", k_min, k_max)

        bic_scores = {}
        sil_scores = {}

        # Sub-sample for silhouette computation if dataset is large
        # (silhouette is O(n²) so we cap at 5000 samples)
        max_sil_samples = min(len(embeddings), 5000)
        sil_indices = np.random.RandomState(config.RANDOM_SEED).choice(
            len(embeddings), max_sil_samples, replace=False
        )
        sil_embeddings = embeddings[sil_indices]

        for k in range(k_min, k_max + 1, k_step):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=config.GMM_COVARIANCE_TYPE,
                random_state=config.RANDOM_SEED,
                n_init=3,        # run 3 initialisations, keep best
                max_iter=200,
            )
            gmm.fit(embeddings)

            # BIC: lower is better
            bic = gmm.bic(embeddings)
            bic_scores[k] = bic

            # Silhouette: higher is better (−1 to 1)
            labels_subset = gmm.predict(sil_embeddings)
            n_unique = len(set(labels_subset))
            if n_unique > 1:
                sil = silhouette_score(sil_embeddings, labels_subset,
                                       sample_size=min(2000, max_sil_samples))
            else:
                sil = -1.0
            sil_scores[k] = sil

            logger.info("  K=%2d → BIC=%.0f, Silhouette=%.4f", k, bic, sil)

        # Select K with lowest BIC
        best_k_bic = min(bic_scores, key=bic_scores.get)

        # Also note K with highest silhouette (for user reference)
        best_k_sil = max(sil_scores, key=sil_scores.get)

        # We prefer BIC as primary criterion because it accounts for
        # model complexity, but we log both for transparency.
        best_k = best_k_bic
        logger.info(
            "Optimal K: %d (BIC-based).  Best Silhouette K: %d.",
            best_k, best_k_sil
        )

        self.n_clusters = best_k

        return {
            "best_k": best_k,
            "bic_scores": bic_scores,
            "sil_scores": sil_scores,
            "best_bic": bic_scores[best_k],
            "best_sil": sil_scores.get(best_k, 0.0),
            "best_k_silhouette": best_k_sil,
        }

    # ─────────────────────────────────────────────────────────
    # Fit & predict
    # ─────────────────────────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit the GMM on document embeddings and store soft assignments.

        After calling this method:
          - self.cluster_probabilities  has shape (N, K)
          - self.hard_labels            has shape (N,)
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters not set. Call find_optimal_k() first.")

        logger.info("Fitting GMM with K=%d…", self.n_clusters)
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=config.GMM_COVARIANCE_TYPE,
            random_state=config.RANDOM_SEED,
            n_init=3,
            max_iter=200,
        )
        self.gmm.fit(embeddings)

        # Soft assignment: probability that each doc belongs to each cluster
        self.cluster_probabilities = self.gmm.predict_proba(embeddings)

        # Hard assignment (for convenience / evaluation)
        self.hard_labels = self.gmm.predict(embeddings)

        logger.info("GMM fit complete.  Cluster sizes: %s",
                     dict(zip(*np.unique(self.hard_labels, return_counts=True))))

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return soft cluster assignments for new embeddings.

        Returns shape (N, K) array of probabilities.
        """
        if self.gmm is None:
            raise ValueError("GMM not fitted yet.  Call fit() first.")
        return self.gmm.predict_proba(embeddings)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return hard cluster assignments for new embeddings.

        Returns shape (N,) array of cluster IDs.
        """
        if self.gmm is None:
            raise ValueError("GMM not fitted yet.  Call fit() first.")
        return self.gmm.predict(embeddings)

    def get_dominant_cluster(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        For a single embedding, return (cluster_id, probability).
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        probs = self.predict_proba(embedding)[0]
        cluster_id = int(np.argmax(probs))
        return cluster_id, float(probs[cluster_id])

    def get_top_clusters(
        self, embedding: np.ndarray, top_n: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Return the top-N clusters for a single embedding, sorted by probability.

        WHY: Semantically similar queries can land in different dominant clusters
        (e.g., "gun laws in America" vs "American firearm legislation" may get
        different dominant clusters due to subtle phrasing differences).  By
        returning the top-N clusters, the cache can search multiple buckets,
        dramatically reducing false misses while keeping lookup cost bounded.

        Returns list of (cluster_id, probability) tuples.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        probs = self.predict_proba(embedding)[0]
        top_indices = np.argsort(probs)[::-1][:top_n]
        return [(int(idx), float(probs[idx])) for idx in top_indices]

    # ─────────────────────────────────────────────────────────
    # Boundary-case analysis
    # ─────────────────────────────────────────────────────────

    def analyze_boundary_cases(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        threshold: float = config.BOUNDARY_UNCERTAINTY_THRESHOLD,
        max_results: int = 20,
    ) -> List[Dict]:
        """
        Identify documents where the model is most uncertain.

        A document is a "boundary case" if its maximum cluster probability
        is below `threshold`.  These documents straddle cluster boundaries
        and often represent genuinely ambiguous or multi-topic content.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, dim)
        documents : List[str]
        threshold : float
            Max probability below which a doc is considered uncertain.
        max_results : int
            Return at most this many boundary cases (sorted by uncertainty).

        Returns
        -------
        List of dicts, each with:
          - index       : document index
          - text        : first 200 chars of the document
          - max_prob    : highest cluster probability
          - top_clusters: list of (cluster_id, probability) for top 3 clusters
        """
        if self.cluster_probabilities is None:
            raise ValueError("Must call fit() before analyzing boundaries.")

        probs = self.cluster_probabilities
        max_probs = probs.max(axis=1)

        # Find documents below the uncertainty threshold
        uncertain_mask = max_probs < threshold
        uncertain_indices = np.where(uncertain_mask)[0]

        # Sort by uncertainty (lowest max_prob first = most uncertain)
        sorted_indices = uncertain_indices[np.argsort(max_probs[uncertain_indices])]

        boundary_cases = []
        for idx in sorted_indices[:max_results]:
            doc_probs = probs[idx]
            top_k_clusters = np.argsort(doc_probs)[::-1][:3]
            top_clusters = [
                {"cluster_id": int(c), "probability": float(doc_probs[c])}
                for c in top_k_clusters
            ]
            boundary_cases.append({
                "index": int(idx),
                "text_preview": documents[idx][:200],
                "max_probability": float(max_probs[idx]),
                "top_clusters": top_clusters,
            })

        logger.info(
            "Boundary analysis: %d / %d documents below threshold %.2f.",
            len(uncertain_indices), len(documents), threshold
        )
        return boundary_cases

    # ─────────────────────────────────────────────────────────
    # Cluster content analysis — "show what lives in them"
    # ─────────────────────────────────────────────────────────

    def show_cluster_contents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        targets: List[str],
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Show the most representative documents in each cluster.

        The brief asks: "convince a sceptical reader that the clusters are
        semantically meaningful.  Show what lives in them."

        For each cluster we:
          1. Find the documents assigned with highest probability.
          2. Show their text previews and ground-truth categories.
          3. Compute the dominant category distribution.

        This provides concrete evidence that clusters are topically coherent.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, dim)
        documents : List[str]
        targets : List[str]   — ground-truth category labels
        top_n : int           — number of representative docs per cluster

        Returns
        -------
        List of dicts (one per cluster), each with:
          - cluster_id       : int
          - size             : int
          - top_categories   : dict {category: count}  (most common labels)
          - representative_docs : list of {text_preview, category, probability}
        """
        if self.cluster_probabilities is None:
            raise ValueError("Must call fit() before showing contents.")

        probs = self.cluster_probabilities  # (N, K)
        results = []

        for k in range(self.n_clusters):
            # Probabilities of all docs belonging to cluster k
            cluster_probs = probs[:, k]

            # Top-N most confident documents for this cluster
            top_indices = np.argsort(cluster_probs)[::-1][:top_n]

            # All documents hard-assigned to this cluster
            hard_members = np.where(self.hard_labels == k)[0]

            # Category distribution within this cluster
            cluster_categories = [targets[i] for i in hard_members]
            from collections import Counter
            cat_counts = dict(Counter(cluster_categories).most_common(5))

            rep_docs = []
            for idx in top_indices:
                rep_docs.append({
                    "text_preview": documents[idx][:200],
                    "category": targets[idx],
                    "probability": float(cluster_probs[idx]),
                })

            results.append({
                "cluster_id": k,
                "size": len(hard_members),
                "top_categories": cat_counts,
                "representative_docs": rep_docs,
            })

        return results

    def get_cluster_summary(self) -> Dict:
        """
        Return a summary of the clustering: sizes, mean probabilities, etc.
        """
        if self.cluster_probabilities is None:
            return {}

        labels, counts = np.unique(self.hard_labels, return_counts=True)
        mean_max_prob = float(self.cluster_probabilities.max(axis=1).mean())

        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": {int(l): int(c) for l, c in zip(labels, counts)},
            "mean_max_probability": mean_max_prob,
            "n_boundary_cases": int(
                (self.cluster_probabilities.max(axis=1)
                 < config.BOUNDARY_UNCERTAINTY_THRESHOLD).sum()
            ),
        }


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data for a quick sanity check
    rng = np.random.RandomState(42)
    fake_embeddings = rng.randn(500, config.EMBEDDING_DIM).astype(np.float32)
    fake_docs = [f"Document {i}" for i in range(500)]

    clusterer = FuzzyClusterer(n_clusters=5)
    clusterer.fit(fake_embeddings)

    summary = clusterer.get_cluster_summary()
    print(f"✓ Cluster summary: {summary}")

    boundaries = clusterer.analyze_boundary_cases(fake_embeddings, fake_docs)
    print(f"✓ Found {len(boundaries)} boundary cases.")
    if boundaries:
        print(f"  Most uncertain: max_prob={boundaries[0]['max_probability']:.3f}")
