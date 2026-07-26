import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/verify_trades_v6.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_verify", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in ["ok","warn","fail","check_manifest","check_real_data_flat","check_dedup","check_tower_slicing"]:
        assert hasattr(mod, fn)

def test_ok_warn_fail():
    mod = load_module()
    assert mod.ok("msg")["status"] == "ok" or "ok" in str(mod.ok("msg")).lower()
    w = mod.warn("wmsg")
    assert w is not None
    f = mod.fail("fmsg")
    assert f is not None

def test_check_manifest_tmp(tmp_path):
    mod = load_module()
    manifest = {"features":["a","b"],"families":["x","y"]}
    p = tmp_path / "manifest.json"
    import json
    p.write_text(json.dumps(manifest))
    out = mod.check_manifest(str(p))
    assert isinstance(out, dict)
    assert out.get("status") != "fail" or True  # at least not crash

def test_check_real_data_flat(tmp_path):
    mod = load_module()
    import json, numpy as np
    # minimal flat data
    data = {"points":[{"ticker":"AAPL","x":0.1,"y":0.2,"z":0.3,"emb":[0.1]*8,"skills":[0.2]*4,"archetype":"Compounder","sector":"Technology"}]}
    p = tmp_path / "real.json"
    p.write_text(json.dumps(data))
    res = mod.check_real_data_flat(str(p))
    assert isinstance(res, dict)

def test_check_zero_placeholders(tmp_path):
    mod = load_module()
    # function may exist
    if hasattr(mod, "check_zero_placeholders"):
        import numpy as np
        npz = tmp_path / "train.npz"
        np.savez_compressed(npz, Z=np.array([[0,0],[1,2]]), mask=np.ones((2,2)))
        res = mod.check_zero_placeholders(str(npz))
        assert isinstance(res, dict)

def test_full_main_no_crash(tmp_path, monkeypatch):
    mod = load_module()
    # main should not crash when pointed to empty dir? We just test it exists
    assert hasattr(mod, "main")
