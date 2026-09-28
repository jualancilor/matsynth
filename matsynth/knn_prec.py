"""Langkah 5: head_prec = retrieval nearest-neighbor di ruang embedding.

Non-parametrik: prediksi precursor set untuk query = precursor set milik
tetangga terdekat (cosine) di antara target train yang punya resep. Metrik:
Jaccard antara set prediksi dan ground truth.
"""

import numpy as np


def _normalize_rows(Z: np.ndarray) -> np.ndarray:
    assert Z.ndim == 2
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    assert (norms > 1e-12).all(), "ada embedding norm-nol"
    return Z / norms


def nearest_neighbors(Z_train: np.ndarray, Z_query: np.ndarray, k: int = 1) -> np.ndarray:
    """Indeks k tetangga terdekat (cosine) di train untuk tiap query. [nq, k]."""
    assert Z_train.shape[1] == Z_query.shape[1]
    assert 1 <= k <= Z_train.shape[0]
    sims = _normalize_rows(Z_query) @ _normalize_rows(Z_train).T  # [nq, nt]
    # sort stabil dengan tie-break indeks agar deterministik
    order = np.lexsort((np.arange(Z_train.shape[0])[None, :].repeat(len(sims), 0), -sims), axis=1)
    idx = order[:, :k]
    assert idx.shape == (Z_query.shape[0], k)
    return idx


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def retrieve_and_score(Z_train: np.ndarray, train_prec_sets: list[frozenset],
                       Z_query: np.ndarray, query_gt_sets: list[list[frozenset]],
                       k: int = 3) -> list[dict]:
    """Untuk tiap query: prediksi = precursor set tetangga #1.

    query_gt_sets: per query BISA lebih dari satu resep ground-truth (satu
    formula sering punya beberapa resep di literatur); skor = max Jaccard
    terhadap salah satu GT. Dilaporkan juga best-of-top-k.
    """
    assert len(train_prec_sets) == Z_train.shape[0]
    assert len(query_gt_sets) == Z_query.shape[0]
    assert all(len(g) >= 1 for g in query_gt_sets), "tiap query butuh >=1 GT set"
    idx = nearest_neighbors(Z_train, Z_query, k=min(k, Z_train.shape[0]))
    results = []
    for qi in range(Z_query.shape[0]):
        pred = train_prec_sets[idx[qi, 0]]
        gts = query_gt_sets[qi]
        topk_j = [max(jaccard(train_prec_sets[j], g) for g in gts) for j in idx[qi]]
        results.append({
            "neighbor_ids": [int(j) for j in idx[qi]],
            "predicted_precursors": sorted(pred),
            "true_precursor_sets": [sorted(g) for g in gts],
            "jaccard_top1": max(jaccard(pred, g) for g in gts),
            "jaccard_best_of_topk": max(topk_j),
            "exact_match_top1": float(any(pred == g for g in gts)),
        })
    return results
