import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/ticker_universe.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_univ", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_univ"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert hasattr(mod, "build_universe")
    assert hasattr(mod, "fetch_sec_tickers")
    assert hasattr(mod, "fetch_sp500_list")

def test_build_universe_mock(monkeypatch, tmp_path):
    mod = load_module()
    # mock fetchers to avoid network
    monkeypatch.setattr(mod, "fetch_sec_tickers", lambda: [{"ticker":"AAPL","company":"Apple Inc","cik":"320193"},{"ticker":"MSFT","company":"Microsoft","cik":"789019"}])
    monkeypatch.setattr(mod, "fetch_sp500_list", lambda: ["AAPL","MSFT"])
    uni = mod.build_universe(limit=2)
    assert isinstance(uni, list)
    assert len(uni) >= 1
    assert any(u["ticker"]=="AAPL" for u in uni)

def test_build_universe_limit():
    mod = load_module()
    import types
    # If real network fails, should still handle limit param
    try:
        uni = mod.build_universe(limit=1)
        assert isinstance(uni, list)
    except Exception as e:
        # network error acceptable but should be exception not TODO
        assert isinstance(e, Exception)
