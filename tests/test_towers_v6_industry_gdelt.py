"""auto-generated test gap mapper for towers_v6.industry_gdelt - coverage <80%"""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "towers_v6"))

try:
    from pipeline.towers_v6 import industry_gdelt as target_module
except Exception:
    try:
        from towers_v6 import industry_gdelt as target_module
    except Exception:
        try:
            import industry_gdelt as target_module
        except Exception:
            target_module = None


@pytest.fixture
def sample_equity_data():
    return {
        "sectors": ["Technology", "Energy", "Financials"],
        "years": list(range(2015, 2025)),
        "tickers": ["AAPL", "XOM", "JPM"],
    }


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(2025)
    return rng.normal(0, 1, size=(11, 8)).astype(np.float32)


@pytest.mark.parametrize("sector", ["Technology", "Energy", "Healthcare"])
def test_towers_v6_industry_gdelt_sector_features(sector, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - sector GDELT feature shape")


@pytest.mark.parametrize("year", [2018, 2020, 2022])
def test_towers_v6_industry_gdelt_year_synth(year, sample_equity_data):
    assert False, "TODO: assert synthetic industry features year prior"


@pytest.mark.parametrize("fallback", [True, False])
def test_towers_v6_industry_gdelt_fallback(fallback, sample_equity_data, sample_embedding):
    pytest.skip("TODO: fill assert - fallback synthetic vs real GDELT")
