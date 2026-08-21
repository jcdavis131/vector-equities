"""Regression coverage for pipeline/composite_score.py.

composite_quality() feeds train_mtnn.py's promotion gate (should_promote,
`cqs >= baseline + 0.005`). A worst-case market_directional_acc of exactly
0.0 (model gets every directional call wrong) must be scored as a real
penalty, not silently treated the same as a missing/untracked metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))

from composite_score import composite_quality, should_promote  # noqa: E402


def _base_report(market_directional_acc):
    return {
        "held_out_recall": {"test": {}},
        "cross_cycle_archetype_purity_at_20": None,
        "next_profile": {},
        "sector_top1_acc": None,
        "market_directional_acc": market_directional_acc,
    }


def test_market_acc_zero_is_penalized_not_ignored():
    """A model that is wrong on every directional call (acc=0.0) must not
    score the same as a model with no market signal reported at all."""
    zero_acc = composite_quality(_base_report(0.0))
    missing_acc = composite_quality(_base_report(None))

    assert zero_acc["parts"]["market_bonus"] == -0.5
    assert missing_acc["parts"]["market_bonus"] == 0.0
    assert zero_acc["cqs"] < missing_acc["cqs"]
    assert zero_acc["cqs"] == 0.0
    assert missing_acc["cqs"] == 0.05


def test_market_bonus_scales_linearly_with_accuracy():
    perfect = composite_quality(_base_report(1.0))
    coin_flip = composite_quality(_base_report(0.5))
    worst = composite_quality(_base_report(0.0))

    assert perfect["parts"]["market_bonus"] == 0.5
    assert coin_flip["parts"]["market_bonus"] == 0.0
    assert worst["parts"]["market_bonus"] == -0.5
    assert perfect["cqs"] > coin_flip["cqs"] > worst["cqs"]


def test_market_bonus_clips_beyond_unit_range():
    # accuracy is a fraction in [0, 1] in practice, but the clip must hold
    # even if an upstream caller passes something out of range.
    over = composite_quality(_base_report(2.0))
    under = composite_quality(_base_report(-1.0))
    assert over["parts"]["market_bonus"] == 0.5
    assert under["parts"]["market_bonus"] == -0.5


def test_recall_and_purity_none_do_not_contribute():
    cq = composite_quality(_base_report(None))
    # neither recall nor purity present -> only the market baseline (0.5*0.10) counts
    assert cq["cqs"] == 0.05
    assert cq["parts"]["recall"] is None
    assert cq["parts"]["purity"] is None


def test_should_promote_gate_blocks_on_zero_market_acc_and_low_recall():
    report = _base_report(0.0)
    report["held_out_recall"] = {"test": {"recall_at_10_mtnn": 0.2}}
    ok, why = should_promote(report, baseline=0.60)
    assert ok is False
    assert "needs improvement" in why


def test_should_promote_gate_passes_on_strong_report():
    report = {
        "held_out_recall": {"test": {"recall_at_10_mtnn": 0.9}},
        "cross_cycle_archetype_purity_at_20": 0.8,
        "next_profile": {"test": {"r2": 0.5}},
        "sector_top1_acc": 0.9,
        "market_directional_acc": 0.7,
    }
    ok, why = should_promote(report, baseline=0.60)
    assert ok is True
    assert "promote" in why
