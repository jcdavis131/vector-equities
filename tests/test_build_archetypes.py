"""auto-generated test gap mapper for build_archetypes - coverage <80%"""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

try:
    from pipeline import build_archetypes as target_module
except Exception:
    try:
        import build_archetypes as target_module
    except Exception:
        target_module = None


@pytest.fixture
def sample_equity_data():
    return {
        "tickers": ["AAPL", "MSFT", "TSLA"],
        "Z": np.random.randn(20, 14).astype(np.float32),
        "sectors": ["Technology", "Technology", "Consumer Discretionary"],
    }


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(7)
    return rng.normal(0, 1, size=(20, 64)).astype(np.float32)


@pytest.mark.parametrize("k", [4, 8, 12])
def test_build_archetypes_cluster_count(k, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - cluster count validation")


@pytest.mark.parametrize("sector", ["Technology", "Financials", "Healthcare"])
def test_build_archetypes_sector_purity(sector, sample_equity_data):
    assert False, "TODO: assert archetype sector purity"


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_build_archetypes_reproducibility(seed, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - kmeans reproducibility")
