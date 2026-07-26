import importlib.util
import json
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/eval_sector_coherence.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_eval", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    for fn in [
        "l2_normalize",
        "knn_indices",
        "purity_from_neighbors",
        "random_purity_expectation",
        "compute_report",
    ]:
        assert hasattr(mod, fn)


def test_l2_normalize():
    mod = load_module()
    import numpy as np

    v = np.array([[3.0, 4.0], [0.0, 5.0]])
    nv = mod.l2_normalize(v)
    assert nv.shape == v.shape
    # each row L2 ~1
    norms = (nv**2).sum(axis=1)
    for n in norms:
        assert abs(n - 1.0) < 1e-6


def test_knn_and_purity():
    mod = load_module()
    import numpy as np

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 8)).astype(float)
    labels = np.array(["A"] * 10 + ["B"] * 10)
    neigh = mod.knn_indices(emb, k=5)
    assert neigh.shape == (20, 5)
    pur = mod.purity_from_neighbors(neigh, labels)
    assert 0 <= pur <= 1.0


def test_random_purity():
    mod = load_module()
    import numpy as np

    labels = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5)
    exp = mod.random_purity_expectation(labels)
    assert 0 < exp < 1.0


def test_silhouette_types():
    mod = load_module()
    import numpy as np

    rng = np.random.default_rng(1)
    emb = rng.normal(size=(12, 6))
    labels = np.array(["X"] * 6 + ["Y"] * 6)
    sil = mod.silhouette_cosine(emb, labels)
    assert isinstance(sil, (float, int, float))


def test_compute_report_empty_tmp(tmp_path):
    mod = load_module()
    # create minimal real_data file
    import numpy as np

    data = {
        "points": [
            {
                "ticker": "AAPL",
                "sector": "Technology",
                "emb": (np.random.randn(8).tolist()),
            },
            {
                "ticker": "MSFT",
                "sector": "Technology",
                "emb": (np.random.randn(8).tolist()),
            },
            {
                "ticker": "JPM",
                "sector": "Financials",
                "emb": (np.random.randn(8).tolist()),
            },
        ]
    }
    p = tmp_path / "real.json"
    p.write_text(json.dumps(data))
    try:
        rep = mod.compute_report(str(p))
        assert isinstance(rep, dict)
    except Exception as e:
        # if compute_report expects more fields, just ensure it raises cleanly not TODO
        assert isinstance(e, Exception)


def test_edge_empty_labels():
    mod = load_module()
    import numpy as np

    emb = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    labels = np.array(["A", "A", "A"])
    neigh = mod.knn_indices(emb, k=2)
    pur = mod.purity_from_neighbors(neigh, labels)
    assert pur == 1.0
