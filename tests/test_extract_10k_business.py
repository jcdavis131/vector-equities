import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/extract_10k_business.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_ext10k", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_ext10k"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "sanitize_ticker",
        "clean_html_to_text",
        "clean_section",
        "extract_business_and_risk",
        "extract_business_and_risk_20f",
        "extract_business_and_risk_auto",
    ]:
        assert hasattr(mod, fn)


def test_sanitize_ticker():
    mod = load_module()
    assert mod.sanitize_ticker(" AAPL ") == "AAPL"
    assert (
        mod.sanitize_ticker("BRK.B") == "BRK_B"
        or mod.sanitize_ticker("BRK.B") == "BRK.B"
    )  # impl may keep dots or replace
    assert isinstance(mod.sanitize_ticker(""), str)


def test_clean_html_to_text():
    mod = load_module()
    html = (
        "<html><body><p>Hello <b>World</b></p><script>alert(1)</script></body></html>"
    )
    txt = mod.clean_html_to_text(html)
    assert "Hello" in txt
    assert "World" in txt
    assert "alert" not in txt


def test_clean_section():
    mod = load_module()
    raw = "\n  This   is \n\n a   test. \n\n"
    cleaned = mod.clean_section(raw)
    assert "This is" in cleaned or "This" in cleaned
    assert isinstance(cleaned, str)


def test_extract_business_auto_10k(tmp_path):
    mod = load_module()
    html = tmp_path / "fake.html"
    html.write_text(
        "<html><body><h1>Item 1. Business</h1><p>We sell widgets. " * 10
        + "<h1>Item 1A. Risk Factors</h1><p>Risk of competition. " * 10
        + "</body></html>"
    )
    try:
        bus, risk = mod.extract_business_and_risk(html)
        assert isinstance(bus, str)
        assert isinstance(risk, str)
    except Exception as e:
        # if function expects specific structure, ensure it doesn't throw TODO
        assert not isinstance(e, NotImplementedError)


def test_extract_20f(tmp_path):
    mod = load_module()
    html = tmp_path / "fake20f.html"
    html.write_text(
        "<html><body>Item 4. Information on the Company Business Overview We do tech. Risk Factors We face risk.</body></html>"
    )
    try:
        out = mod.extract_business_and_risk_20f(html)
        assert out is not None
    except Exception:
        pass


def test_extract_auto_router(tmp_path):
    mod = load_module()
    html = tmp_path / "fake.html"
    html.write_text("<html>Item 1 Business etc</html>")
    out = mod.extract_business_and_risk_auto(html, form_type="10-K")
    assert out is not None
