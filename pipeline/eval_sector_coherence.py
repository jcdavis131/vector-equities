"""Sector-coherence eval for the company embedding space.

Engineering quality metric for the 64-d MTNN embedding: how coherent is the
embedding geometry with respect to GICS sector labels? This measures the
embedding space only — it is NOT predictive of returns and NOT investment
advice.

Metrics
-------
- knn_sector_purity_at_10: fraction of each row's 10 nearest neighbors
  (cosine) sharing its sector label, averaged over rows.
- knn_sector_purity_at_10_cross_ticker: same, but neighbors from the same
  ticker are excluded. Training uses same-ticker adjacent-FY contrastive
  pairs, so same-ticker neighbors share a sector trivially; this variant
  removes that inflation.
- silhouette_cosine: sklearn silhouette score of sector clusters, cosine
  distance, on the full evaluated matrix.

Baseline (named)
----------------
Random-assignment expectation given sector sizes: the repo carries no raw
description text locally (pipeline/data/chunks_v2 holds an index only), so a
TF-IDF baseline is not computable from the repo. Purity baseline is the
closed-form expectation sum_s (n_s/n) * ((n_s-1)/(n-1)) plus an empirical
label-permutation baseline (fixed seed); the silhouette baseline is the
label-permutation mean (theoretical expectation ~0).

Output: assets/eval_sector_coherence.json

Usage: python pipeline/eval_sector_coherence.py
"""

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"

K = 10
N_PERMUTATIONS = 5
PERMUTATION_SEED = 0

# Canonical sector labels carried by assets/real_data.json. Rows whose label
# is not in this list (e.g. the "Inc." name-parse artifact) are excluded and
# counted in the report.
CANONICAL_SECTORS = frozenset(
    {
        "Technology",
        "Healthcare",
        "Financials",
        "Consumer Discretionary",
        "Consumer Staples",
        "Industrials",
        "Energy",
        "Materials",
        "Utilities",
        "Real Estate",
        "Communication",
    }
)


def load_points(path):
    """Load embeddings + labels from real_data.json, dropping bad sectors."""
    data = json.loads(Path(path).read_text())
    points = data["points"]
    keep = [p for p in points if p.get("sector") in CANONICAL_SECTORS]
    excluded = len(points) - len(keep)
    emb = np.asarray([p["emb"] for p in keep], dtype=np.float32)
    sectors = np.asarray([p["sector"] for p in keep])
    tickers = np.asarray([p["ticker"] for p in keep])
    meta = {
        "model": data.get("model", "unknown"),
        "dim": int(emb.shape[1]),
        "n_rows_total": len(points),
        "n_rows_excluded_bad_sector": excluded,
    }
    return emb, sectors, tickers, meta


def l2_normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def knn_indices(emb, k=K, tickers=None, block=512):
    """Top-k cosine neighbor indices per row (self excluded).

    If tickers is given, rows sharing the query row's ticker are excluded
    (cross-ticker variant).
    """
    e = l2_normalize(emb)
    n = e.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = e[start:stop] @ e.T
        for i in range(start, stop):
            row = sims[i - start]
            row[i] = -np.inf
            if tickers is not None:
                row[tickers == tickers[i]] = -np.inf
            top = np.argpartition(row, -k)[-k:]
            out[i] = top[np.argsort(row[top])[::-1]]
    return out


def purity_from_neighbors(neighbors, labels):
    """Mean fraction of neighbors sharing the query row's label."""
    return float((labels[neighbors] == labels[:, None]).mean())


def random_purity_expectation(labels):
    """Closed-form E[purity] under random neighbor assignment.

    For a row in sector s, a uniformly random other row shares its sector
    with probability (n_s - 1) / (n - 1); weight by sector frequencies.
    """
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    return float(sum((c / n) * ((c - 1) / (n - 1)) for c in counts))


def permutation_purity(neighbors, labels, n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED):
    """Empirical purity baseline: mean purity over label permutations."""
    rng = np.random.default_rng(seed)
    scores = [
        purity_from_neighbors(neighbors, rng.permutation(labels)) for _ in range(n_perm)
    ]
    return float(np.mean(scores))


def silhouette_cosine(emb, labels):
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(emb, labels, metric="cosine"))


def permutation_silhouette(emb, labels, n_perm=3, seed=PERMUTATION_SEED):
    rng = np.random.default_rng(seed)
    scores = [silhouette_cosine(emb, rng.permutation(labels)) for _ in range(n_perm)]
    return float(np.mean(scores))


def compute_report(real_data_path):
    emb, sectors, tickers, meta = load_points(real_data_path)
    n = emb.shape[0]
    sector_names, sector_counts = np.unique(sectors, return_counts=True)
    print(f"Evaluating {n} rows, {len(sector_names)} sectors, dim {meta['dim']}")

    t0 = time.time()
    nn_all = knn_indices(emb, k=K)
    purity = purity_from_neighbors(nn_all, sectors)
    nn_cross = knn_indices(emb, k=K, tickers=tickers)
    purity_cross = purity_from_neighbors(nn_cross, sectors)
    print(
        f"purity@{K} {purity:.4f} cross-ticker {purity_cross:.4f} "
        f"({time.time() - t0:.1f}s)"
    )

    baseline_purity = random_purity_expectation(sectors)
    baseline_perm = permutation_purity(nn_all, sectors)

    t0 = time.time()
    sil = silhouette_cosine(emb, sectors)
    sil_perm = permutation_silhouette(emb, sectors)
    print(
        f"silhouette {sil:.4f} perm-baseline {sil_perm:.4f} ({time.time() - t0:.1f}s)"
    )

    return {
        "eval": "sector_coherence",
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_source": "assets/real_data.json",
        "embedding_model": meta["model"],
        "dim": meta["dim"],
        "n_rows_total": meta["n_rows_total"],
        "n_rows_evaluated": n,
        "n_rows_excluded_bad_sector": meta["n_rows_excluded_bad_sector"],
        "n_companies": len(np.unique(tickers)),
        "n_sectors": len(sector_names),
        "sector_sizes": {
            s: int(c)
            for s, c in zip(sector_names.tolist(), sector_counts.tolist(), strict=True)
        },
        "baseline": {
            "name": "random-assignment expectation given sector sizes",
            "why": (
                "repo carries no raw description text locally "
                "(pipeline/data/chunks_v2 is an index only), so a TF-IDF text "
                "baseline is not computable; closed-form random expectation "
                "plus seeded label-permutation used instead"
            ),
            "permutation_seed": PERMUTATION_SEED,
            "n_permutations": N_PERMUTATIONS,
        },
        "metrics": {
            f"knn_sector_purity_at_{K}": {
                "definition": (
                    f"fraction of each row's {K} nearest neighbors (cosine) "
                    "sharing its GICS sector, averaged over rows; same-ticker "
                    "neighbors allowed"
                ),
                "score": round(purity, 4),
                "baseline_random_assignment": round(baseline_purity, 4),
                "baseline_label_permutation": round(baseline_perm, 4),
                "lift_over_random": round(purity / baseline_purity, 2),
            },
            f"knn_sector_purity_at_{K}_cross_ticker": {
                "definition": (
                    "same purity but neighbors from the query row's own ticker "
                    "are excluded; removes the trivial same-ticker inflation "
                    "from contrastive training"
                ),
                "score": round(purity_cross, 4),
                "baseline_random_assignment": round(baseline_purity, 4),
                "lift_over_random": round(purity_cross / baseline_purity, 2),
            },
            "silhouette_cosine": {
                "definition": (
                    "sklearn silhouette score of sector clusters, cosine "
                    "distance, full evaluated matrix; range [-1, 1], 0 is "
                    "chance-level"
                ),
                "score": round(sil, 4),
                "baseline_label_permutation": round(sil_perm, 4),
            },
        },
        "provenance": (
            "Measured on the published matrix as served. Note: the "
            "2026-07-20 S&P 500 expansion (pipeline/expand_sp500.py, commit "
            "7d93c0b) filled rows for newly added tickers with "
            "sector-centroid + Gaussian-noise placeholder embeddings rather "
            "than model outputs; diagnostic runs show model-derived and "
            "placeholder subsets score similarly on this eval."
        ),
        "note": (
            "Engineering quality metric for the embedding space only; "
            "measures label coherence of the geometry, not investment merit "
            "and not predictive of returns."
        ),
    }


def main():
    report = compute_report(ASSETS_DIR / "real_data.json")
    out_path = ASSETS_DIR / "eval_sector_coherence.json"
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
