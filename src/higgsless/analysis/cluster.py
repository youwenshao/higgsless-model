"""Average-linkage clustering of unit vectors by cosine similarity (small n, no sklearn needed)."""

from __future__ import annotations

import numpy as np


def cluster(vectors: np.ndarray, threshold: float) -> list[int]:
    """Return a cluster label per row. Clusters merge while their average similarity >= threshold."""
    n = len(vectors)
    if n == 0:
        return []
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sim = v @ v.T
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                score = float(sim[np.ix_(clusters[a], clusters[b])].mean())
                if score > best:
                    best, pair = score, (a, b)
        if pair is None or best < threshold:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    # Biggest cluster first so P1 is the most frequent performer.
    clusters.sort(key=lambda c: (-len(c), min(c)))
    labels = [0] * n
    for label, members in enumerate(clusters):
        for i in members:
            labels[i] = label
    return labels
