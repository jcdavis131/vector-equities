"""auto-generated test gap mapper for composite_score - coverage <80%"""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

try:
    from pipeline import composite_score as target_module
except Exception:
    try:
        import composite_score as target_module
    except Exception:
        target_module = None


@pytest.fixture
def sample_equity_data():
    return {
        "report": {
            "held_out_recall": {"test": {"recall_at_10_mtnn": 0.42}},
            "cross_cycle_archetype_purity_at_20": 0.61,
            "next_profile": {"test": {"r2": 0.15}},
            "sector_accuracy": 0.55,
            "market_bonus": 0.1,
        }
    }


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(123)
    return rng.normal(0, 1, size=(64,)).astype(np.float32)


@pytest.mark.parametrize("recall", [0.1, 0.42, 0.9])
def test_composite_score_recall_weighting(recall, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - CQS recall weighting")


@pytest.mark.parametrize("r2", [-0.2, 0.0, 0.35])
def test_composite_score_r2_clipping(r2, sample_equity_data):
    assert False, "TODO: assert R2 clipped to [0,1] in CQS"


@pytest.mark.parametrize("purity", [0.2, 0.5, 0.95])
def test_composite_score_purity_contrib(purity, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - purity contribution to composite")
