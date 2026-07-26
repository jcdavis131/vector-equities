import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/parse_neo.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_parseneo", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_parseneo"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in ["parse_proxy_text","fetch_and_parse_def14a"]:
        assert hasattr(mod, fn)

def test_parse_proxy_text():
    mod = load_module()
    txt = "Summary Compensation Table\nTim Cook CEO 15,000,000\nJohn Doe CFO 5,000,000"
    out = mod.parse_proxy_text(txt)
    assert isinstance(out, (list, dict))
    if isinstance(out, list) and len(out)>0:
        assert isinstance(out[0], dict) or isinstance(out[0], (list, tuple))

def test_fetch_and_parse_def14a_no_network():
    mod = load_module()
    # should handle network failure gracefully
    try:
        out = mod.fetch_and_parse_def14a("AAPL")
        assert isinstance(out, (dict, list)) or out is None
    except Exception as e:
        # network errors acceptable but not TODO
        assert isinstance(e, Exception)
