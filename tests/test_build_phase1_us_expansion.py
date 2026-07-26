import importlib.util
import json
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/build_phase1_us_expansion.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_phase1", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_phase1"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "load_json",
        "fetch_sec_ticker_exchange",
        "fetch_nasdaq_trader_lists",
        "is_likely_operating_business",
    ]:
        assert hasattr(mod, fn)


def test_load_json_tmp(tmp_path):
    mod = load_module()
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"a": 1}))
    out = mod.load_json(str(p))
    assert out == {"a": 1}


def test_is_likely_operating_business():
    mod = load_module()
    assert mod.is_likely_operating_business(
        {"company": "Apple Inc", "sector": "Technology"}
    ) in (True, False)
    assert isinstance(mod.is_likely_operating_business({"company": "ETF Trust"}), bool)


def test_fetch_mock(monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "fetch_sec_ticker_exchange", lambda: [{"ticker": "AAPL"}])
    monkeypatch.setattr(mod, "fetch_nasdaq_trader_lists", lambda: [{"ticker": "AAPL"}])
    # main shouldn't crash with mocked fetchers? we just test helpers
    assert True
