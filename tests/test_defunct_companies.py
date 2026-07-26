import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/defunct_companies.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_defunct", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_defunct"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke_funcs():
    mod = load_module()
    for fn in [
        "is_known_defunct",
        "get_defunct_info",
        "parse_date",
        "is_stale_trading",
        "is_defunct_ticker",
        "classify_status",
        "list_defunct_universe",
        "detect_via_sec_filings",
    ]:
        assert hasattr(mod, fn)


def test_is_known_defunct_true():
    mod = load_module()
    assert mod.is_known_defunct("ENRON") is True
    assert mod.is_known_defunct("LEH") is True
    assert mod.is_known_defunct("AAPL") is False


def test_alias_lookup():
    mod = load_module()
    # ENRNQ is alias of ENRON -> should be defunct
    assert mod.is_known_defunct("ENRNQ") is True
    info = mod.get_defunct_info("ENRNQ")
    assert info is not None
    assert "Enron" in info["company"]


def test_get_defunct_info_none():
    mod = load_module()
    assert mod.get_defunct_info("AAPL") is None
    assert mod.get_defunct_info("") is None


def test_parse_date():
    mod = load_module()
    d = mod.parse_date("2023-03-09")
    assert d is not None
    assert d.year == 2023
    assert mod.parse_date("") is None
    assert mod.parse_date("not-a-date") is None


def test_is_stale_trading():
    mod = load_module()
    # very old date should be stale >730 days
    assert mod.is_stale_trading("2015-01-01", stale_days=730) is True
    # today not stale
    import datetime

    today = datetime.date.today().isoformat()
    assert mod.is_stale_trading(today, stale_days=730) is False


def test_is_defunct_ticker_heuristic():
    mod = load_module()
    # known defunct always True even without dates
    assert mod.is_defunct_ticker("ENRON") is True
    # unknown with old last close
    assert mod.is_defunct_ticker("FAKE", last_close_date_str="2010-01-01") is True
    # fresh ticker not defunct
    import datetime

    today = datetime.date.today().isoformat()
    assert mod.is_defunct_ticker("AAPL", last_close_date_str=today) is False


def test_classify_status_fields():
    mod = load_module()
    res = mod.classify_status(
        "LEH", last_close_date_str="2008-09-12", first_close_date_str="1970-01-01"
    )
    assert res["is_defunct"] is True
    assert res["ticker"] == "LEH"
    assert "category" in res
    assert res["era"] in ("1960s", "1970s", "1980s", "1990s", "2000s", "2010s+")


def test_list_defunct_universe_length():
    mod = load_module()
    uni = mod.list_defunct_universe()
    assert isinstance(uni, list)
    assert len(uni) >= 50  # we hardcode ~80
    for entry in uni:
        assert "ticker" in entry and "is_defunct" in entry


def test_detect_via_sec_filings():
    mod = load_module()
    import datetime

    old = "2010-01-01"
    assert mod.detect_via_sec_filings("12345", old, gap_years=4) is True
    today = datetime.date.today().isoformat()
    assert mod.detect_via_sec_filings("12345", today, gap_years=4) is False
    assert mod.detect_via_sec_filings("12345", None) is False
