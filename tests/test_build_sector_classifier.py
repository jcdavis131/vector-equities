import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/build_sector_classifier.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_sect", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_sect"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "normalize_sector",
        "load_universe",
        "get_chunk_text",
        "build_training_data",
        "train_classifier",
        "save_model",
    ]:
        assert hasattr(mod, fn)


def test_normalize_sector_various():
    mod = load_module()
    assert mod.normalize_sector("Tech") == "Technology"
    assert mod.normalize_sector("Health Care") == "Health Care"
    assert mod.normalize_sector("") == "Misc"
    assert (
        mod.normalize_sector("N/A") == "Misc" or mod.normalize_sector("n/a") == "Misc"
    )
    assert mod.normalize_sector("Basic Industries") in (
        "Materials",
        "Industrials",
        "Materials",
    )  # mapped


def test_normalize_sector_fuzzy():
    mod = load_module()
    assert mod.normalize_sector("information technology services") == "Technology"


def test_get_chunk_text_missing(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    out = mod.get_chunk_text("FAKE")
    assert isinstance(out, str)
    assert out == ""
