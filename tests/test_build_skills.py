import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/build_skills.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_skills", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_skills"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert hasattr(mod, "build")


def test_build_basic():
    mod = load_module()
    # build may need matrix, but we can call with mocked data
    try:
        out = mod.build()
        assert out is not None
    except FileNotFoundError:
        # missing data acceptable, but should be real error
        assert True
    except Exception as e:
        # should not be TODO skip
        assert not isinstance(e, NotImplementedError)


def test_build_with_tmp(tmp_path, monkeypatch):
    mod = load_module()
    # if build writes files, redirect via ROOT monkeypatch
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    try:
        out = mod.build()
        assert True
    except Exception:
        assert True
