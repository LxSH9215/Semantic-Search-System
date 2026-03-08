"""
data_loader.py — Load & Clean the 20 Newsgroups Dataset from UCI Repository
============================================================================
Design decisions for the cleaning pipeline (justified below each step):

The 20 Newsgroups dataset is *notoriously noisy* for NLP tasks.  Raw posts
contain email headers, quoted replies, signatures, and other boilerplate that
would pollute embeddings.  Our goal is to isolate the **semantic body text**
so that the embedding model can focus on topical content.

Data Source:
    UCI Machine Learning Repository (ID 113)
    https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
    We use the `ucimlrepo` package as the primary loader.  If unavailable,
    we fall back to `sklearn.datasets.fetch_20newsgroups` which ships the
    same underlying corpus.
"""

import re
import logging
from typing import List, Tuple, Optional

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Cleaning helpers
# ─────────────────────────────────────────────────────────────

def _strip_email_headers(text: str) -> str:
    """
    Remove RFC-822–style email headers (From:, Subject:, Lines:, etc.).

    WHY: Headers contain metadata (author emails, dates, message IDs) that
    carry no topical semantics.  Worse, unique email addresses would create
    spurious "topics" in the embedding space, hurting clustering quality.

    HOW: We look for the first blank line that traditionally separates headers
    from the body in Usenet posts.  Everything before it is dropped.
    """
    # Find the first double-newline (header/body separator in email/NNTP)
    header_end = text.find("\n\n")
    if header_end != -1:
        # Heuristic: only strip if the block before looks like headers
        # (contains lines with ":" in the first 40 chars)
        candidate = text[:header_end]
        header_like_lines = sum(
            1 for line in candidate.split("\n")
            if ":" in line[:40]
        )
        if header_like_lines >= 2:
            return text[header_end + 2:]
    return text


def _strip_quoted_replies(text: str) -> str:
    """
    Remove lines that are quoted replies (lines starting with > or |).

    WHY: Quoted text is a duplicate of content already present in other
    documents.  Including it would (a) inflate document lengths unevenly,
    (b) bias embeddings toward popular threads, and (c) cause the same
    content to cluster together artificially.
    """
    lines = text.split("\n")
    cleaned = [
        line for line in lines
        if not re.match(r"^\s*[>|]", line)
    ]
    return "\n".join(cleaned)


def _strip_signatures(text: str) -> str:
    """
    Remove email signatures (text after the standard '-- ' delimiter).

    WHY: Signatures contain personal info, ASCII art, disclaimers, and
    witty quotes — none of which relate to the newsgroup topic.  The
    Usenet convention is "-- \\n" (dash-dash-space-newline) as the
    signature separator.
    """
    # Standard Usenet signature delimiter
    sig_marker = "\n-- \n"
    idx = text.find(sig_marker)
    if idx != -1:
        return text[:idx]
    return text


def _strip_footers(text: str) -> str:
    """
    Remove common footer patterns found in newsgroup posts.

    WHY: Footers like "Send stripping instructions to …" or university
    disclaimers are boilerplate that appears across many posts and would
    create artificial similarity between unrelated documents.
    """
    # Common footer patterns
    footer_patterns = [
        r"--+\s*$",                          # lines of dashes at end
        r"^\s*_{3,}\s*$",                     # lines of underscores
        r"(?i)disclaimer\s*:",                # "Disclaimer:" blocks
        r"(?i)this message was sent",         # auto-generated footers
    ]
    lines = text.split("\n")
    # Walk backwards and drop lines matching footer patterns
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if any(re.search(pat, last) for pat in footer_patterns):
            lines.pop()
        else:
            break
    return "\n".join(lines)


def _normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into single spaces / newlines.

    WHY: After stripping headers, quotes, and signatures we often end up
    with runs of blank lines.  Normalising keeps document lengths meaningful
    and avoids wasting embedding model context on whitespace tokens.
    """
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs within a line
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_document(text: str) -> str:
    """
    Full cleaning pipeline applied to each raw newsgroup post.

    The order matters:
      1. Strip headers first  (they can contain ">" characters that look like quotes)
      2. Strip quoted replies  (before signature removal, since quotes can contain "-- ")
      3. Strip signatures
      4. Strip footers
      5. Normalize whitespace  (final cosmetic pass)
    """
    text = _strip_email_headers(text)
    text = _strip_quoted_replies(text)
    text = _strip_signatures(text)
    text = _strip_footers(text)
    text = _normalize_whitespace(text)
    return text


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────

def _load_from_ucimlrepo() -> Optional[Tuple[List[str], List[str]]]:
    """
    Primary loader: use the official `ucimlrepo` package.

    Returns (documents, target_names) or None on failure.
    """
    try:
        from ucimlrepo import fetch_ucirepo

        logger.info("Fetching 20 Newsgroups from UCI ML Repository (id=%d)…",
                     config.UCI_DATASET_ID)
        dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)

        # The dataset features contain the text data
        X = dataset.data.features
        y = dataset.data.targets

        # Extract text column — the column name may vary
        text_col = X.columns[0]  # typically the first (and only) feature column
        documents = X[text_col].tolist()

        # Extract target names if available
        if y is not None and len(y.columns) > 0:
            target_col = y.columns[0]
            target_names = y[target_col].tolist()
        else:
            target_names = ["unknown"] * len(documents)

        logger.info("Loaded %d documents from UCI repository.", len(documents))
        return documents, target_names

    except Exception as e:
        logger.warning("ucimlrepo failed (%s), falling back to sklearn.", e)
        return None


def _load_from_sklearn() -> Tuple[List[str], List[str]]:
    """
    Fallback loader: use sklearn's bundled copy of the same corpus.

    We set remove=() to get the RAW text (headers included) so our own
    cleaning pipeline can handle everything consistently.
    """
    from sklearn.datasets import fetch_20newsgroups

    logger.info("Fetching 20 Newsgroups from sklearn (fallback)…")
    dataset = fetch_20newsgroups(subset="all", remove=())
    documents = dataset.data
    target_names = [dataset.target_names[t] for t in dataset.target]
    logger.info("Loaded %d documents from sklearn.", len(documents))
    return documents, target_names


def load_and_clean() -> Tuple[List[str], List[str], List[int]]:
    """
    Load the 20 Newsgroups dataset and apply the full cleaning pipeline.

    Returns
    -------
    cleaned_docs : List[str]
        Cleaned document texts (only docs longer than MIN_DOC_LENGTH).
    target_names : List[str]
        Corresponding category labels (for evaluation / analysis only).
    original_indices : List[int]
        Indices into the original dataset (useful for traceability).
    """
    # Try UCI first, fall back to sklearn
    result = _load_from_ucimlrepo()
    if result is None:
        result = _load_from_sklearn()

    raw_docs, raw_targets = result

    cleaned_docs = []
    target_names = []
    original_indices = []

    for idx, (doc, target) in enumerate(zip(raw_docs, raw_targets)):
        if not isinstance(doc, str):
            continue
        cleaned = clean_document(doc)
        # Filter out documents that are too short after cleaning.
        # WHY: Very short documents (< MIN_DOC_LENGTH chars) are usually
        # empty replies, signatures-only, or corrupt entries.  They produce
        # near-zero-information embeddings that hurt clustering quality.
        if len(cleaned) >= config.MIN_DOC_LENGTH:
            cleaned_docs.append(cleaned)
            target_names.append(str(target))
            original_indices.append(idx)

    logger.info(
        "Cleaning complete: %d → %d documents (dropped %d short docs).",
        len(raw_docs), len(cleaned_docs), len(raw_docs) - len(cleaned_docs)
    )
    return cleaned_docs, target_names, original_indices


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs, targets, indices = load_and_clean()
    print(f"\n✓ Loaded {len(docs)} cleaned documents.")
    print(f"  Example (first 200 chars): {docs[0][:200]}…")
    print(f"  Category: {targets[0]}")
