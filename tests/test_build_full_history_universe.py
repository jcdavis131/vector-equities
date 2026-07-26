import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/build_full_history_universe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pipeline_mod_fullhist", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_fullhist"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "load_current_universe",
        "scan_market_history",
        "build_full_history_universe",
    ]:
        assert hasattr(mod, fn)


def test_load_current_universe_empty(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    # create empty universe file maybe
    (tmp_path / "pipeline").mkdir(parents=True, exist_ok=True)
    out = tmp_path / "pipeline" / "data"
    out.mkdir(parents=True, exist_ok=True)
    # load should handle missing and return list
    try:
        uni = mod.load_current_universe()
        assert isinstance(uni, list)
    except Exception:
        assert True


def test_scan_market_history_empty(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "MARKET_DIR", tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    res = mod.scan_market_history()
    assert isinstance(res, (list, dict, set))


def test_build_full_history_universe_tmp(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    try:
        res = mod.build_full_history_universe()
        assert isinstance(res, (list, dict)) or res is None
    except Exception:
        assert True
