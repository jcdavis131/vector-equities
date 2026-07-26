import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/build_revised_towers.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_rev", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_rev"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in ["stable_hash","hash_text_to_vec","parse_company_file","build_company_index","build_towers_for_matrix"]:
        assert hasattr(mod, fn)

def test_stable_hash_deterministic():
    mod = load_module()
    h1 = mod.stable_hash("hello world")
    h2 = mod.stable_hash("hello world")
    h3 = mod.stable_hash("different")
    assert h1 == h2
    assert h1 != h3
    assert isinstance(h1, int)

def test_hash_text_to_vec():
    mod = load_module()
    vec = mod.hash_text_to_vec("Apple builds iPhones and sells cloud services", dim=8)
    assert len(vec) == 8
    # normalized L2 ~1
    import math
    norm = math.sqrt(sum(v*v for v in vec))
    assert abs(norm-1.0) < 1e-5 or norm==0
    empty = mod.hash_text_to_vec("", dim=8)
    assert empty == [0.0]*8
    short = mod.hash_text_to_vec("hi", dim=8)
    assert short == [0.0]*8  # len<20 returns zeros per impl

def test_parse_company_file(tmp_path):
    mod = load_module()
    p = tmp_path / "company_AAPL.md"
    p.write_text("# Apple Inc (AAPL) — Technology\n\n**Ticker:** AAPL | **CIK:** 320193 | **Sector:** Technology | **Has Business:** True\n\n**Industry:** Consumer Electronics\n\nmarketcap=$2500B\n\nHas Financials: True\n\n## Business Model\nWe build iPhones, iPads and services.\n\n## Risk Factors\nHighly competitive market, supply chain risks.\n\nReportable segments consist of Americas 40%, Europe 30%, Greater China 20%.")
    info = mod.parse_company_file(p)
    assert info is not None
    assert info["ticker"] == "AAPL"
    assert "Technology" in info["sector"] or "technology" in info["sector"].lower()
    assert info["marketcap"] == 2500.0
    assert len(info["business"]) > 20

def test_build_towers_small(tmp_path, monkeypatch):
    mod = load_module()
    import numpy as np, json
    # minimal matrix
    N=3
    Z = np.random.randn(N,2).astype(np.float32)
    mask = np.ones_like(Z)
    tickers = np.array(["AAPL","MSFT","GOOGL"])
    npz_path = tmp_path / "train_matrix.npz"
    np.savez_compressed(npz_path, Z=Z, mask=mask, ticker=tickers)
    manifest = {"features":["F1","F2"],"families":["a","b"]}
    man_path = tmp_path / "feature_manifest.json"
    man_path.write_text(json.dumps(manifest))
    # fake company index
    company_index = {
        "AAPL": {"ticker":"AAPL","sector":"Technology","industry":"Consumer Electronics","marketcap":2000.0,"has_fin":True,"has_bus":True,"has_risk":True,"business":"iPhone business model with services and recurring revenue from App Store, iCloud, and subscriptions spanning global markets with strong moat and ecosystem lock-in for long term growth","risk":"Competitive risk and regulation from global antitrust and supply chain disruptions across Asia and geopolitical tensions","geo":"Americas 45%, Europe 30%, Greater China 15%, Japan 5%, Rest of Asia Pacific"},
        "MSFT": {"ticker":"MSFT","sector":"Technology","industry":"Software","marketcap":2500.0,"has_fin":True,"has_bus":True,"has_risk":False,"business":"Cloud Azure and Office productivity suite with enterprise licensing and AI Copilot integration driving enterprise cloud adoption worldwide","risk":"","geo":"North America, Europe, Asia Pacific"},
        "GOOGL": {"ticker":"GOOGL","sector":"Technology","industry":"Internet Services","marketcap":1500.0,"has_fin":False,"has_bus":True,"has_risk":False,"business":"Search advertising, YouTube, Cloud Platform and Waymo autonomous driving with extensive data network effects and machine learning capabilities","risk":"","geo":""},
    }
    out_mat, out_man, shape, cnt, hits = mod.build_towers_for_matrix(npz_path, man_path, company_index, out_suffix="testrev")
    assert out_mat.exists()
    assert out_man.exists()
    assert shape[0]==N
    assert shape[1]== 2+44  # 2 old + 44 new
    assert hits["business"]>=1
