"""Tests for pipeline/eval_sector_coherence.py.

Unit-tests the metric math on synthetic geometry, then gates the emitted
assets/eval_sector_coherence.json against schema, ranges, and the live
assets/real_data.json row counts.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from eval_sector_coherence import (  # noqa: E402
    CANONICAL_SECTORS,
    K,
    knn_indices,
    l2_normalize,
    permutation_purity,
    purity_from_neighbors,
    random_purity_expectation,
    silhouette_cosine,
)

EVAL_PATH = ROOT / "assets" / "eval_sector_coherence.json"
REAL_DATA_PATH = ROOT / "assets" / "real_data.json"


def _synthetic_clusters(n_per=40, dim=16, n_clusters=3, sep=8.0, seed=7):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(n_clusters, dim)) * sep
    emb = np.concatenate(
        [c + rng.normal(0, 0.3, size=(n_per, dim)) for c in centers]
    ).astype(np.float32)
    labels = np.repeat([f"c{i}" for i in range(n_clusters)], n_per)
    return emb, labels


def test_purity_perfect_on_separated_clusters():
    emb, labels = _synthetic_clusters()
    nn = knn_indices(emb, k=K)
    assert purity_from_neighbors(nn, labels) == pytest.approx(1.0)


def test_purity_near_chance_on_shuffled_labels():
    emb, labels = _synthetic_clusters()
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(labels)
    nn = knn_indices(emb, k=K)
    expected = random_purity_expectation(shuffled)
    assert purity_from_neighbors(nn, shuffled) == pytest.approx(expected, abs=0.08)


def test_random_expectation_matches_permutation_baseline():
    emb, labels = _synthetic_clusters()
    nn = knn_indices(emb, k=K)
    closed_form = random_purity_expectation(labels)
    empirical = permutation_purity(nn, labels, n_perm=20, seed=1)
    assert empirical == pytest.approx(closed_form, abs=0.05)


def test_cross_ticker_exclusion():
    emb, _labels = _synthetic_clusters(n_per=12, n_clusters=2)
    tickers = np.repeat([f"t{i}" for i in range(6)], 4)
    nn = knn_indices(emb, k=3, tickers=tickers)
    for i, row in enumerate(nn):
        assert all(tickers[j] != tickers[i] for j in row)


def test_silhouette_positive_on_separated_clusters():
    emb, labels = _synthetic_clusters()
    assert silhouette_cosine(emb, labels) > 0.5


def test_l2_normalize_unit_norm():
    emb, _ = _synthetic_clusters(n_per=5, n_clusters=2)
    norms = np.linalg.norm(l2_normalize(emb), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_emitted_report_schema_and_consistency():
    assert EVAL_PATH.exists(), "run pipeline/eval_sector_coherence.py first"
    report = json.loads(EVAL_PATH.read_text())
    assert report["eval"] == "sector_coherence"
    assert report["embedding_source"] == "assets/real_data.json"
    for key in ("computed_at", "baseline", "provenance", "note"):
        assert key in report
    m = report["metrics"]
    purity = m[f"knn_sector_purity_at_{K}"]
    cross = m[f"knn_sector_purity_at_{K}_cross_ticker"]
    sil = m["silhouette_cosine"]
    assert 0.0 <= purity["score"] <= 1.0
    assert 0.0 <= cross["score"] <= 1.0
    assert 0.0 < purity["baseline_random_assignment"] < 1.0
    assert -1.0 <= sil["score"] <= 1.0
    assert report["n_sectors"] == len(report["sector_sizes"])
    assert set(report["sector_sizes"]) <= CANONICAL_SECTORS
    assert sum(report["sector_sizes"].values()) == report["n_rows_evaluated"]
    assert (
        report["n_rows_evaluated"] + report["n_rows_excluded_bad_sector"]
        == report["n_rows_total"]
    )


def test_emitted_report_matches_live_asset_counts():
    report = json.loads(EVAL_PATH.read_text())
    data = json.loads(REAL_DATA_PATH.read_text())
    points = data["points"]
    assert report["n_rows_total"] == len(points)
    kept = [p for p in points if p.get("sector") in CANONICAL_SECTORS]
    assert report["n_rows_evaluated"] == len(kept)
    assert report["n_companies"] == len({p["ticker"] for p in kept})
    assert report["dim"] == len(kept[0]["emb"])
