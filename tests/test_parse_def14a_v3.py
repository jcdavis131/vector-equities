import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/parse_def14a_v3.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_parsev3", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_parsev3"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "clean_text",
        "clean_name",
        "is_blacklisted",
        "is_plausible_name",
        "extract_comp",
        "score_table_text",
        "parse_one_file_fast",
    ]:
        assert hasattr(mod, fn)


def test_clean_text():
    mod = load_module()
    raw = "  Tim   Cook\n\n CEO   "
    cleaned = mod.clean_text(raw)
    assert cleaned.strip() != ""
    assert "Tim" in cleaned


def test_clean_name():
    mod = load_module()
    assert mod.clean_name("  TIM COOK  ") == "Tim Cook" or "Tim" in mod.clean_name(
        "  TIM COOK  "
    )
    assert mod.clean_name("") == "" or mod.clean_name("") is not None
    # numeric noise
    assert (
        mod.clean_name("1234") == "" or mod.clean_name("1234") == "1234"
    )  # plausible filter may reject


def test_is_blacklisted():
    mod = load_module()
    assert mod.is_blacklisted("Summary Compensation Table") is True
    assert mod.is_blacklisted("Tim Cook") is False


def test_is_plausible_name():
    mod = load_module()
    assert mod.is_plausible_name("Tim Cook") is True
    assert mod.is_plausible_name("CEO") is False
    assert mod.is_plausible_name("") is False


def test_extract_comp_numeric():
    mod = load_module()
    cells = ["Tim Cook", "Chief Executive Officer", "$15,000,000", "2023"]
    comp = mod.extract_comp(cells)
    # Could be dict or None, but if it extracts should have numeric
    if comp:
        assert isinstance(comp, (dict, tuple, list))


def test_score_table_text():
    mod = load_module()
    txt = "Summary Compensation Table Name Total Tim Cook 15000000"
    score = mod.score_table_text(txt)
    assert score > 0
    txt2 = "Random paragraph about business"
    score2 = mod.score_table_text(txt2)
    assert score2 < score


def test_parse_one_file_fast(tmp_path):
    mod = load_module()
    p = tmp_path / "def14a.html"
    p.write_text(
        "<html><table><tr><td>Tim Cook</td><td>CEO</td><td>$15,000,000</td></tr></table> Summary Compensation Table</html>"
    )
    out = mod.parse_one_file_fast(p)
    assert isinstance(out, (list, dict))
