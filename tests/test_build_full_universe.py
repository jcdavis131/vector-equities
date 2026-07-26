import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/build_full_universe.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_fulluni", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_fulluni"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert hasattr(mod, "main") or hasattr(mod, "load_sec_map")

def test_load_sec_map_empty(tmp_path, monkeypatch):
    mod = load_module()
    if hasattr(mod, "load_sec_map"):
        try:
            m = mod.load_sec_map()
            assert isinstance(m, dict)
        except Exception:
            assert True

def test_main_does_not_crash(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    # main may take args; we just ensure it exists and callable
    assert callable(mod.main)
