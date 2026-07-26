import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/build_exec_registry_v3.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_execreg", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_execreg"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in ["normalize_full","normalize_firstlast","load_parsed","main"]:
        assert hasattr(mod, fn)

def test_normalize_full():
    mod = load_module()
    assert mod.normalize_full("  Tim   Cook ") == "tim cook"
    assert mod.normalize_full("") == ""
    assert mod.normalize_full("JOHN DOE") == "john doe"

def test_normalize_firstlast():
    mod = load_module()
    assert mod.normalize_firstlast("Tim Cook") == ("tim","cook")
    assert mod.normalize_firstlast("Madonna") == ("madonna","") or len(mod.normalize_firstlast("Madonna"))==2
    assert mod.normalize_firstlast("") == ("","")

def test_load_parsed_empty(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    res = mod.load_parsed()
    assert isinstance(res, (list, dict))
