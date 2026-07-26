"""auto-generated test gap mapper for build_advanced_features - coverage <80%"""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

try:
    from pipeline import build_advanced_features as target_module
except Exception:
    try:
        import build_advanced_features as target_module
    except Exception:
        target_module = None


@pytest.fixture
def sample_equity_data():
    return {
        "ticker": "AAPL",
        "sector": "Technology",
        "fiscal_year": 2023,
        "features": np.random.randn(14).astype(np.float32),
        "market_cap": 2.5e12,
    }


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, size=(64,)).astype(np.float32)


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL"])
def test_build_advanced_features_tower_build(ticker, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - implement advanced tower build check")


@pytest.mark.parametrize("family", ["profitability", "growth", "defense"])
def test_build_advanced_features_family_metrics(family, sample_equity_data):
    assert False, "TODO: assert family metrics computed"


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_build_advanced_features_horizon_logic(horizon, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - horizon embedding logic")
