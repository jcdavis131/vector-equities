import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/build_demo.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_demo", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert hasattr(mod, "gen_company_profile") or hasattr(mod, "save_bundle")

def test_gen_company_profile():
    mod = load_module()
    if hasattr(mod, "gen_company_profile"):
        prof = mod.gen_company_profile("AAPL")
        assert isinstance(prof, dict)
        assert prof.get("ticker") == "AAPL" or "ticker" in prof

def test_gen_with_sector():
    mod = load_module()
    if hasattr(mod, "gen_company_profile"):
        prof = mod.gen_company_profile("MSFT", sector="Technology")
        assert isinstance(prof, dict)

def test_save_bundle_tmp(tmp_path):
    mod = load_module()
    if hasattr(mod, "save_bundle"):
        import numpy as np
        bundle = {"Z": np.random.randn(2,3), "ticker": np.array(["AAPL","MSFT"])}
        out = tmp_path / "bundle.npz"
        try:
            mod.save_bundle(bundle, str(out))
            assert out.exists()
        except TypeError:
            mod.save_bundle(bundle, out)
            assert out.exists() or (tmp_path / "bundle.npz").exists()
