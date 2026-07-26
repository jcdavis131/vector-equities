import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/feature_spec.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pipeline_mod_featspec", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_featspec"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert (
        hasattr(mod, "FEATURE_FAMILIES")
        or hasattr(mod, "ALL_FEATURES")
        or hasattr(mod, "FAMILY_OF")
    )


def test_families_structure():
    mod = load_module()
    if hasattr(mod, "FEATURE_FAMILIES"):
        fams = mod.FEATURE_FAMILIES
        assert isinstance(fams, dict)
        assert len(fams) >= 10
        for fam, feats in fams.items():
            assert isinstance(feats, list)
            assert len(feats) >= 2


def test_all_features_unique():
    mod = load_module()
    if hasattr(mod, "ALL_FEATURES"):
        af = mod.ALL_FEATURES
        assert (
            len(af) == len(set(af))
        )  # unique? might have duplicates like GROSS_MARGIN appears twice across families
        assert len(af) >= 50


def test_game_profile_features():
    mod = load_module()
    if hasattr(mod, "GAME_PROFILE_FEATURES"):
        gpf = mod.GAME_PROFILE_FEATURES
        assert len(gpf) == 14 or len(gpf) >= 10
        assert all(isinstance(s, str) for s in gpf)


def test_skill_keys():
    mod = load_module()
    if hasattr(mod, "SKILL_KEYS"):
        sk = mod.SKILL_KEYS
        assert len(sk) == 12
