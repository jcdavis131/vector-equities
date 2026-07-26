"""auto-generated test gap mapper for eval_v6_real - coverage <80%"""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

try:
    from pipeline import eval_v6_real as target_module
except Exception:
    try:
        import eval_v6_real as target_module
    except Exception:
        target_module = None


@pytest.fixture
def sample_equity_data():
    return {
        "points": [
            {"ticker": f"T{i:04d}", "sector": "Technology", "fiscal_year": 2022}
            for i in range(10)
        ],
        "tickers": [f"T{i:04d}" for i in range(10)],
    }


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(99)
    return rng.normal(0, 1, size=(10, 64)).astype(np.float32)


@pytest.mark.parametrize("k", [5, 10, 20])
def test_eval_v6_real_knn_purity(k, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - recall/purity knn eval")


@pytest.mark.parametrize("sector", ["Technology", "Energy", "Financials"])
def test_eval_v6_real_sector_coverage(sector, sample_equity_data):
    assert False, "TODO: assert sector evaluation coverage"


@pytest.mark.parametrize("horizon", [1, 3, 7])
def test_eval_v6_real_horizon_eval(horizon, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - forward horizon eval")
