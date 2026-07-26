import importlib.util
import pathlib
import sys


def load_module():
    mod_path = pathlib.Path(
        "/home/hatch/workspace/vector-equities/pipeline/expanding_window_cv.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_mod_ewcv", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_mod_ewcv"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke():
    mod = load_module()
    assert hasattr(mod, "ExpandingWindowCV")
    assert hasattr(mod, "FoldSpec")
    assert hasattr(mod, "aggregate_metrics")


def test_to_year_parsing():
    mod = load_module()
    assert mod._to_year(2021) == 2021
    assert mod._to_year("FY2021") == 2021
    assert mod._to_year("1996-97") == 1996
    assert mod._to_year("2023-24") == 2023


def test_unique_sorted_years():
    mod = load_module()
    out = mod._unique_sorted_years([2023, 2021, 2022, 2021])
    assert out == [2021, 2022, 2023]


def test_get_folds_basic():
    mod = load_module()
    cv = mod.ExpandingWindowCV(
        years=list(range(2015, 2025)), min_train_years=5, val_years=1, step=1
    )
    folds = cv.get_folds()
    assert len(folds) > 0
    # first fold train 2015-2019 val 2020
    assert folds[0].train_years == [2015, 2016, 2017, 2018, 2019]
    assert folds[0].val_years == [2020]
    # no overlap train/val and expanding
    for f in folds:
        assert max(f.train_years) < min(f.val_years)


def test_split_sample_years():
    mod = load_module()
    sample_years = [2015, 2015, 2016, 2017, 2018, 2019, 2020, 2020, 2021]
    cv = mod.ExpandingWindowCV(
        years=range(2015, 2023), min_train_years=3, val_years=1, step=1
    )
    splits = list(cv.split(sample_years))
    assert len(splits) > 0
    for tr, val, spec in splits:
        assert len(tr) > 0 and len(val) > 0
        # train idx years < val idx years
        tr_years = [sample_years[i] for i in tr]
        val_years = [sample_years[i] for i in val]
        assert max(tr_years) < min(val_years)


def test_aggregate_metrics():
    mod = load_module()
    metrics = [{"recall_at_10": 0.8}, {"recall_at_10": 0.9}, {"recall_at_10": 0.85}]
    agg = mod.aggregate_metrics(metrics)
    assert agg["n_folds"] == 3
    assert "recall_at_10" in agg
    assert abs(agg["recall_at_10"]["mean"] - 0.85) < 1e-6


def test_aggregate_empty():
    mod = load_module()
    assert mod.aggregate_metrics([])["n_folds"] == 0


def test_temporal_stability():
    mod = load_module()
    cv = mod.ExpandingWindowCV(
        years=range(2015, 2025), min_train_years=5, val_years=1, step=1
    )
    folds = cv.get_folds()
    # fake metrics trending up
    metrics = [{"recall_at_10": 0.5 + i * 0.05} for i in range(len(folds))]
    rep = mod.temporal_stability_report(folds, metrics, metric_key="recall_at_10")
    assert rep["metric"] == "recall_at_10"
    assert "slope_per_year" in rep


def test_expanding_window_indices_helper():
    mod = load_module()
    splits = mod.expanding_window_indices(
        sample_years=list(range(2015, 2025)) * 2, min_train_years=5, val_years=1, step=2
    )
    assert len(splits) > 0
