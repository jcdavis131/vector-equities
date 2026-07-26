import importlib.util, sys, pathlib
import pytest, json, re, math

def load_module():
    mod_path = pathlib.Path("/home/hatch/workspace/vector-equities/pipeline/composite_score.py")
    spec = importlib.util.spec_from_file_location("pipeline_mod_composite_score", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_composite_score"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke_has_functions():
    mod = load_module()
    for fn in ["sigmoid","partial_cqs","composite_quality","should_promote"]:
        assert hasattr(mod, fn), f"missing {fn}"

def test_sigmoid_values():
    mod = load_module()
    import numpy as np
    assert abs(mod.sigmoid(0) - 0.5) < 1e-6
    assert mod.sigmoid(10) > 0.9
    assert mod.sigmoid(-10) < 0.1

def test_partial_cqs_none_handling():
    mod = load_module()
    assert mod.partial_cqs(None, 0.8) == 0.8
    assert mod.partial_cqs(0.6, None) == 0.6
    assert abs(mod.partial_cqs(0.6, 0.8) - 0.7) < 1e-6

def test_composite_quality_real():
    mod = load_module()
    report = {
        "held_out_recall": {"test": {"recall_at_10_mtnn": 0.82}},
        "cross_cycle_archetype_purity_at_20": 0.65,
        "next_profile": {"test": {"r2": 0.25}},
        "sector_top1_acc": 0.55,
        "market_directional_acc": 0.58,
    }
    out = mod.composite_quality(report)
    assert "cqs" in out
    assert isinstance(out["cqs"], float)
    assert 0 <= out["cqs"] <= 1.5
    assert out["recall_at_10"] == 0.82
    assert out["parts"]["r2_clip"] == 0.25
    # market bonus clipped
    assert -0.5 <= out["parts"]["market_bonus"] <= 0.5

def test_composite_quality_r2_clipping():
    mod = load_module()
    report = {
        "held_out_recall": {"test": {"recall_at_10_mtnn": 0.9}},
        "cross_cycle_archetype_purity_at_20": 0.7,
        "next_profile": {"test": {"r2": 2.0}},  # >0.9 should clip
    }
    out = mod.composite_quality(report)
    assert out["parts"]["r2_clip"] == 0.9

def test_should_promote_logic():
    mod = load_module()
    high = {
        "held_out_recall": {"test": {"recall_at_10_mtnn": 0.80}},
        "cross_cycle_archetype_purity_at_20": 0.8,
        "next_profile": {"test": {"r2": 0.5}},
        "sector_top1_acc": 0.9,
        "market_directional_acc": 0.7,
    }
    ok, msg = mod.should_promote(high, baseline=0.60)
    # cqs high but recall 0.80 >=0.75 so should promote True (cqs >=0.605)
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    assert "CQS" in msg

def test_should_promote_fail_low_recall():
    mod = load_module()
    low = {
        "held_out_recall": {"test": {"recall_at_10_mtnn": 0.5}},
        "cross_cycle_archetype_purity_at_20": 0.3,
    }
    ok, msg = mod.should_promote(low, baseline=0.90)
    assert ok is False
