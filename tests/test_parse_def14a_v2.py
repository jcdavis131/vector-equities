import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/parse_def14a_v2.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_parsev2", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_parsev2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert hasattr(mod, "parse_one")

def test_parse_one_tmp(tmp_path):
    mod = load_module()
    p = tmp_path / "def14a.html"
    p.write_text("<html><body>CEO Tim Cook Age 62 Total 15M</body></html>")
    out = mod.parse_one(str(p))
    assert isinstance(out, (dict, list))

def test_parse_one_missing():
    mod = load_module()
    try:
        out = mod.parse_one("/nonexistent/path.html")
        assert isinstance(out, (dict, list))
    except Exception as e:
        assert isinstance(e, Exception) and not isinstance(e, NotImplementedError)
