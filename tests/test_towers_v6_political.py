"""auto-generated test gap mapper for towers_v6.political - coverage <80%"""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "towers_v6"))

try:
    from pipeline.towers_v6 import political as target_module
except Exception:
    try:
        from towers_v6 import political as target_module
    except Exception:
        try:
            import political as target_module
        except Exception:
            target_module = None


@pytest.fixture
def sample_equity_data():
    return {
        "years": list(range(2015, 2025)),
        "election_years": [2016, 2020, 2024],
        "tickers": ["AAPL", "BA", "LMT"],
    }


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(321)
    return rng.normal(0, 1, size=(10, 6)).astype(np.float32)


@pytest.mark.parametrize("year", [2016, 2020, 2024])
def test_towers_v6_political_election_proximity(year, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - election proximity scoring")


@pytest.mark.parametrize("gpr_level", ["low", "mid", "high"])
def test_towers_v6_political_gpr_synthetic(gpr_level, sample_equity_data):
    assert False, "TODO: assert GPR synthetic respects priors"


@pytest.mark.parametrize("epu_factor", [0.8, 1.0, 1.5])
def test_towers_v6_political_epu_tariff(epu_factor, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - EPU/tariff tower blend")
