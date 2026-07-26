import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/feature_spec_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pipeline_mod_featspecv2", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_featspecv2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert len(dir(mod)) > 3


def test_has_families_or_features():
    mod = load_module()
    # v2 should have similar structure
    has_fam = (
        hasattr(mod, "FEATURE_FAMILIES")
        or hasattr(mod, "FAMILIES")
        or hasattr(mod, "FEATURES")
    )
    assert has_fam or hasattr(mod, "ALL_FEATURES")
