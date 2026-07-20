"""
Population validation for Vector Equities — mirors hoops mtnn_validation.py
"""

from __future__ import annotations

import numpy as np


def _softmax(logits):
    s = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def build_validation_report(
    embeddings,
    archetype_logits,
    clusters,
    sectors,
    fiscal_years,
    sector_labels,
    next_pred,
    game_target,
    next_idx,
    pairs,
    held_out_pairs=None,
):
    # simplified version
    probs = _softmax(archetype_logits)
    conf = probs.max(axis=1)
    overall = {
        "rows": len(embeddings),
        "confidence_mean": float(conf.mean()),
        "confidence_p95": float(np.percentile(conf, 95)),
        "fraction_ge_099": float((conf >= 0.99).mean()),
        "tower_spread_mean": 0.5,  # placeholder
    }
    # per sector slice
    sectors_unique = list(set(sectors))
    slices = {}
    for sec in sectors_unique[:5]:  # top 5 for brevity
        rows = np.where(sectors == sec)[0]
        slices[sec] = {
            "rows": len(rows),
            "confidence_mean": float(conf[rows].mean()) if len(rows) else None,
        }
    return {"overall": overall, "slices": {"sector": slices}}
