# Real-data multi-target benchmark (vector-bench)

End-to-end, seeded, CPU-only pipeline that wires REAL forward-shifted labels for
the three equities registry targets and runs the
[vector-bench](https://github.com/jcdavis131/vector-hub) multi-target baseline
gauntlet against the repo's real MTNN (`pipeline/model.py` `EquitiesMTNN`).

## Result (fy2022-2024 test, temporal split, seed 42)

| target | primary metric | best baseline | MTNN | verdict |
|---|---|---|---|---|
| forward_return | spearman_ic | pca_ridge(n=16) **0.1481** | 0.0926 | baseline wins |
| forward_realized_vol | spearman_ic | hist_gbm **0.7178** | 0.6965 | baseline wins |
| drawdown_exceedance | roc_auc | ridge **0.6864** | 0.6724 | baseline wins |

The multi-task thesis LOSES on all three targets in this domain. Full details in
`benchmark_report.json` (schema 1.1) and `data/datasheet.json`.

## Reproduce

```bash
pip install -e <vector-hub>/packages/vector-core -e <vector-hub>/packages/vector-bench
python bench/fetch_prices.py --out-dir /tmp/equities_prices   # ~500 Yahoo chart fetches
python bench/build_dataset.py --prices-dir /tmp/equities_prices
python bench/run_benchmark.py
```

Everything is deterministic given the fetched prices (seed 42, torch CPU
threads pinned); label windows end by Jan 2026, so re-fetching does not move
them. See the module docstrings of `build_dataset.py` / `run_benchmark.py` for
the leakage rules (point-in-time feature audit, July-1 anchor, train-only
preprocessing, purged temporal split).
