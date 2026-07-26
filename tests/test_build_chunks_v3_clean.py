import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/build_chunks_v3_clean.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_chunks", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_chunks"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "normalize_sector",
        "tokenize",
        "shingles",
        "jaccard",
        "dedup_paragraphs",
        "clean_paragraph",
        "is_boilerplate",
        "split_into_chunks",
    ]:
        assert hasattr(mod, fn)


def test_normalize_sector():
    mod = load_module()
    assert mod.normalize_sector("Technology") == "Technology"
    assert mod.normalize_sector("Health Care") == "Health Care"
    assert mod.normalize_sector("") == "Misc"
    assert mod.normalize_sector("Tech") == "Technology"


def test_tokenize_shingles_jaccard():
    mod = load_module()
    toks = mod.tokenize("Hello world! Hello")
    assert "hello" in toks and "world" in toks
    sh = mod.shingles(toks, n=2)
    assert isinstance(sh, set) and len(sh) > 0
    a = {"a b c", "d e f"}
    b = {"a b c"}
    assert mod.jaccard(a, b) > 0
    assert mod.jaccard(set(), set()) == 0.0


def test_dedup_paragraphs():
    mod = load_module()
    paras = [
        "This is a unique paragraph about Apple Inc business model.",
        "This is a unique paragraph about Apple Inc business model.",  # duplicate
        "Short",  # too short filtered
        "Another distinct paragraph about cloud services growth strategy and moat.",
    ]
    out = mod.dedup_paragraphs(paras, threshold=0.9)
    assert len(out) <= len(paras)
    # duplicate should be removed (keep first)
    assert any("Apple" in p for p in out)


def test_clean_paragraph():
    mod = load_module()
    raw = "Apple Inc. | 2024 Form 10-K | 24\n\nTable of Contents\n\nReal content here."
    cleaned = mod.clean_paragraph(raw)
    assert "Apple Inc. |" not in cleaned
    assert "Real content" in cleaned


def test_is_boilerplate():
    mod = load_module()
    boiler = "The markets for the company's products and services are highly competitive, there can be no assurance."
    assert mod.is_boilerplate(boiler) is True
    assert mod.is_boilerplate("Our CEO is Tim Cook and we sell iPhones.") is False


def test_split_into_chunks():
    mod = load_module()
    text = (
        "Sentence one. " * 10
        + "\n\n"
        + "Sentence two. " * 10
        + "\n\n"
        + "Sentence three. " * 20
    )
    chunks = mod.split_into_chunks(text, target_tokens=15, overlap=5)
    assert len(chunks) >= 2
    for ch in chunks:
        assert isinstance(ch, str) and len(ch) > 0


def test_process_ticker_no_text(monkeypatch):
    mod = load_module()
    # stub load_v2_chunks and load_extracted_text to force no_text path
    monkeypatch.setattr(mod, "load_v2_chunks", lambda ticker: [])
    monkeypatch.setattr(mod, "load_extracted_text", lambda ticker: ("", ""))
    res = mod.process_ticker(
        {"ticker": "FAKE", "sector": "Technology", "company": "Fake Co"}
    )
    assert res["status"] == "no_text"


def test_process_ticker_with_text(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "CHUNKS_V3_DIR", tmp_path / "chunks_v3")
    monkeypatch.setattr(mod, "WIKI_V3_DIR", tmp_path / "wiki_v3")
    (tmp_path / "chunks_v3").mkdir()
    (tmp_path / "wiki_v3").mkdir()
    long_text = (
        "Business overview of company selling tech gadgets and providing cloud infrastructure services to enterprise customers worldwide with strong moat and recurring revenue. "
        * 12
    )
    # ensure >50 tokens distinct chunks
    fake_chunks = [
        {
            "text": long_text
            + " Segment A unique details about iPhone sales and services growth across geographies."
        },
        {
            "text": long_text
            + " Segment B unique details about Mac, iPad, wearables and supply chain resilience and R&D."
        },
        {
            "text": long_text
            + " Segment C unique details about competitive landscape, regulatory risk and management quality."
        },
    ]
    monkeypatch.setattr(mod, "load_v2_chunks", lambda ticker: fake_chunks)
    monkeypatch.setattr(mod, "load_extracted_text", lambda ticker: ("", ""))
    res = mod.process_ticker(
        {
            "ticker": "AAPL",
            "sector": "Technology",
            "company": "Apple Inc",
            "cik": "320193",
        }
    )
    assert res["status"] == "ok"
    assert res["chunks"] > 0
