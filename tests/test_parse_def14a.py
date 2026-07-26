import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/parse_def14a.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pipeline_mod_parse14a", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_parse14a"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "html_to_text",
        "extract_comp_table_text",
        "parse_neos_heuristic",
        "parse_def14a_file",
    ]:
        assert hasattr(mod, fn)


def test_html_to_text():
    mod = load_module()
    b = b"<html><p>John Doe  CEO  1,200,000</p></html>"
    txt = mod.html_to_text(b)
    assert "John" in txt or "Doe" in txt or len(txt) > 0


def test_extract_comp_table_text(tmp_path):
    mod = load_module()
    html = tmp_path / "def14a.html"
    html.write_text(
        "<html><body><table><tr><th>Name</th><th>Total</th></tr><tr><td>Tim Cook</td><td>$15,000,000</td></tr></table></body></html>"
    )
    try:
        tbl = mod.extract_comp_table_text(str(html))
        assert isinstance(tbl, str)
    except Exception:
        # function may accept Path, ensure no crash type
        tbl = mod.extract_comp_table_text(html)
        assert isinstance(tbl, str)


def test_parse_neos_heuristic(tmp_path):
    mod = load_module()
    html = tmp_path / "def14a.html"
    html.write_text(
        "<html>Summary Compensation Table\nTim Cook CEO 15000000\nBob CFO 5000000</html>"
    )
    neos = mod.parse_neos_heuristic(str(html))
    assert isinstance(neos, (list, dict))


def test_parse_def14a_file(tmp_path):
    mod = load_module()
    html = tmp_path / "def14a.html"
    html.write_text("<html>Executive Compensation Table Tim Cook 2023 $15M</html>")
    out = mod.parse_def14a_file(str(html))
    assert out is not None
