import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/build_demo_v3.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_demov3", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_demov3"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert hasattr(mod, "gen_company_profile")


def test_gen_profile_realistic():
    mod = load_module()
    prof = mod.gen_company_profile("TSLA", sector="Consumer Discretionary")
    assert isinstance(prof, dict)
    # should have some fields
    assert len(prof) >= 2
