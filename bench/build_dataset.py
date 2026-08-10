"""Build the real-data multi-target benchmark dataset for vector-equities.

Features
--------
The (ticker, fiscal-year) feature panel is the repo's own committed real-data
matrix: git blob ``02d107b4f01cb6ecfab8ee0b162d440f42638eae``
(``pipeline/data/train_matrix_real.npz`` as of commit ``dc47078``, the last
commit that tracked it before large build artifacts were gitignored), together
with its manifest ``c6b5c2d:pipeline/data/feature_manifest.json``. It carries
4831 rows x 118 features for 500 tickers, fiscal years 2015-2024, built from
SEC EDGAR XBRL fundamentals plus a per-fiscal-year-end market price cache.

Point-in-time audit (enforced here, not assumed)
------------------------------------------------
Two classes of features exist in that matrix:

1. Historical per-FY features (fundamentals, market_price, valuation, macro).
   The market/valuation features were verified point-in-time against
   independently fetched Yahoo daily history: recomputing RET_12M / RET_6M /
   VOL_252D / PE as of the last trading day <= Dec 31 of the fiscal year
   reproduces the stored values with Spearman rho = 1.000. They are anchored
   at the fiscal-year-end calendar date.
2. Snapshot features copied to every year of a ticker (ownership,
   disclosure_text, sector_context, most management_neo, several form flags,
   CREDIT_SPREAD_PROXY, CASH_CONVERSION_CYCLE). These were fetched once at
   build time (~2026) and therefore CONTAIN FUTURE INFORMATION relative to
   early fiscal years. They are dropped by a deterministic audit: any feature
   whose raw value is constant across every ticker's years (const_frac >=
   0.999 over tickers with >= 3 observations) is excluded.

Labels (strictly forward-shifted)
---------------------------------
Anchor t(row) = first trading day >= July 1 of (fiscal_year + 1). Every
feature uses data <= fiscal-year-end (<= ~Jan 31, fy+1 for the latest fiscal
year ends) and every 10-K for fiscal year fy is filed months before July 1,
fy+1, so features are strictly available before the anchor. Labels use ONLY
prices strictly after the anchor:

- forward_return        = adjclose[t+126] / adjclose[t] - 1   (~6 months)
- forward_realized_vol  = std(daily simple returns over (t, t+126]) * sqrt(252)
- forward_max_drawdown  = min over the window of (price - running peak)/peak,
  computed on adjclose[t+1 .. t+126]; the binary drawdown_exceedance target is
  thresholded at benchmark time against the TRAIN-SPLIT median so no test
  label statistics leak into label construction.

Prices are daily adjusted closes from the Yahoo Finance v8 chart API (the
repo's documented market source; see pipeline/fetch_market_history.py),
fetched by bench/fetch_prices.py. Rows whose ticker has no usable price
history around the window (delisted/renamed symbols) get label_mask = 0 and
are excluded per target, never imputed.

Split (temporal, leakage-safe)
------------------------------
time_key = fiscal year. Harness split: train fy 2015-2021, test fy 2022-2024
(time_cut = 2022). The MTNN additionally holds out fy 2021 from its own
gradient steps as an early-stopping validation year. The latest training
label window ends ~Jan 2023 (fy2021 anchor Jul 2022 + 126 trading days),
strictly before the earliest test anchor (Jul 2023), so no training label
overlaps any test label window.

Usage
-----
    python bench/build_dataset.py --prices-dir <dir from fetch_prices.py> \
        --out bench/data/equities_bench_v1.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

MATRIX_BLOB = "02d107b4f01cb6ecfab8ee0b162d440f42638eae"
MATRIX_BLOB_SHA256 = "60a82d4503e2d698bef4060b1ecb4f817ecd607d3c0cb557b42af1930ed5a6f8"
MANIFEST_REV = "c6b5c2d:pipeline/data/feature_manifest.json"

TRAIN_YEARS = (2015, 2020)  # MTNN gradient years
VAL_YEAR = 2021  # MTNN early-stopping year (part of harness train)
TEST_CUT = 2022  # harness temporal cut: test = fy >= 2022
HORIZON_TDAYS = 126  # ~6 months of trading days
CONST_FRAC_DROP = 0.999


def git_blob(sha: str) -> bytes:
    return subprocess.run(["git", "cat-file", "blob", sha], cwd=REPO, capture_output=True, check=True).stdout


def git_show(rev_path: str) -> bytes:
    return subprocess.run(["git", "show", rev_path], cwd=REPO, capture_output=True, check=True).stdout


def load_panel():
    raw = git_blob(MATRIX_BLOB)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MATRIX_BLOB_SHA256:
        raise RuntimeError(f"matrix blob sha256 mismatch: {digest}")
    import io

    z = np.load(io.BytesIO(raw), allow_pickle=True)
    manifest = json.loads(git_show(MANIFEST_REV))
    return z, manifest


def snapshot_audit(Zr: np.ndarray, tickers: np.ndarray) -> np.ndarray:
    """Fraction of tickers (>=3 obs) whose value never changes, per feature."""
    idx_by_t: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tickers):
        idx_by_t[t].append(i)
    const_frac = np.zeros(Zr.shape[1])
    for j in range(Zr.shape[1]):
        n_const = n_tick = 0
        for idxs in idx_by_t.values():
            if len(idxs) < 3:
                continue
            v = Zr[idxs, j]
            v = v[~np.isnan(v)]
            if len(v) < 3:
                continue
            n_tick += 1
            if np.all(v == v[0]):
                n_const += 1
        const_frac[j] = n_const / max(n_tick, 1)
    return const_frac


def load_prices(prices_dir: Path, ticker: str):
    fp = prices_dir / f"{ticker}.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    dates = np.array(d["dates"])
    adj = np.array([np.nan if v is None else v for v in d["adjclose"]], dtype=float)
    ok = ~np.isnan(adj) & (adj > 0)
    if ok.sum() < 260:
        return None
    return dates[ok], adj[ok]


def forward_labels(dates: np.ndarray, adj: np.ndarray, fy: int):
    """(forward_return, forward_realized_vol, forward_max_drawdown) or None."""
    anchor = np.searchsorted(dates, f"{fy + 1}-07-01", side="left")
    if anchor + HORIZON_TDAYS >= len(adj):
        return None
    # guard: the anchor must actually be near July fy+1 (not a post-gap date)
    if not dates[anchor].startswith(f"{fy + 1}-07"):
        return None
    c0 = adj[anchor]
    window = adj[anchor : anchor + HORIZON_TDAYS + 1]  # t .. t+126 inclusive
    fwd_ret = float(window[-1] / c0 - 1.0)
    rets = np.diff(window) / window[:-1]
    fwd_vol = float(np.std(rets) * np.sqrt(252.0))
    future = window[1:]
    peak = np.maximum.accumulate(future)
    fwd_dd = float(np.min((future - peak) / peak))
    return fwd_ret, fwd_vol, fwd_dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "data" / "equities_bench_v1.npz")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    z, manifest = load_panel()
    feats = manifest["features"]
    families = manifest["families"]
    Zr = z["Z_raw"].astype(np.float64)
    mask = z["mask"].astype(np.float32)
    tickers = z["ticker"].astype(str)
    fy = z["fiscal_year"].astype(int)
    sectors = z["sector"].astype(str)

    # ---- point-in-time feature audit: drop static snapshot features ---------
    const_frac = snapshot_audit(Zr, tickers)
    keep = const_frac < CONST_FRAC_DROP
    dropped = [(feats[j], families[j], round(float(const_frac[j]), 3)) for j in np.where(~keep)[0]]
    kept_feats = [feats[j] for j in np.where(keep)[0]]
    kept_fams = [families[j] for j in np.where(keep)[0]]
    X_raw = Zr[:, keep].astype(np.float32)
    X_mask = mask[:, keep]
    print(f"features kept {keep.sum()}/{len(feats)}; dropped snapshot features:")
    for name, fam, cf in dropped:
        print(f"  - {name} ({fam}, const_frac={cf})")

    # ---- forward labels from real daily prices ------------------------------
    y_ret = np.full(len(fy), np.nan, dtype=np.float32)
    y_vol = np.full(len(fy), np.nan, dtype=np.float32)
    y_dd = np.full(len(fy), np.nan, dtype=np.float32)
    cache: dict[str, object] = {}
    for i in range(len(fy)):
        t = tickers[i]
        if t not in cache:
            cache[t] = load_prices(args.prices_dir, t)
        pr = cache[t]
        if pr is None:
            continue
        lab = forward_labels(pr[0], pr[1], int(fy[i]))
        if lab is None:
            continue
        y_ret[i], y_vol[i], y_dd[i] = lab

    m_ret = ~np.isnan(y_ret)
    m_vol = ~np.isnan(y_vol)
    m_dd = ~np.isnan(y_dd)

    # ---- temporal split indices ---------------------------------------------
    train_idx = np.where(fy <= TRAIN_YEARS[1])[0]
    val_idx = np.where(fy == VAL_YEAR)[0]
    test_idx = np.where(fy >= TEST_CUT)[0]

    ticker_names, ticker_ids = np.unique(tickers, return_inverse=True)

    np.savez_compressed(
        args.out,
        X=X_raw,
        X_mask=X_mask,
        feature_names=np.array(kept_feats),
        feature_families=np.array(kept_fams),
        y_forward_return=y_ret,
        y_forward_realized_vol=y_vol,
        forward_max_drawdown=y_dd,
        mask_forward_return=m_ret,
        mask_forward_realized_vol=m_vol,
        mask_drawdown_exceedance=m_dd,
        entity_id=ticker_ids,
        entity_names=ticker_names,
        ticker=tickers,
        time_key=fy,
        sector=sectors,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        test_cut=np.array([TEST_CUT]),
        horizon_tdays=np.array([HORIZON_TDAYS]),
    )
    n_priced = sum(1 for v in cache.values() if v is not None)
    sheet = {
        "dataset": "equities_bench_v1",
        "rows": int(len(fy)),
        "entities": int(len(ticker_names)),
        "entities_with_prices": int(n_priced),
        "time_range": [int(fy.min()), int(fy.max())],
        "features": {
            "count": int(keep.sum()),
            "source": (
                "repo git blob pipeline/data/train_matrix_real.npz "
                f"({MATRIX_BLOB}, sha256 {MATRIX_BLOB_SHA256}); SEC EDGAR XBRL "
                "fundamentals + per-fiscal-year-end market/valuation features "
                "(verified point-in-time vs independent Yahoo daily history, "
                "Spearman 1.000 on RET_12M/RET_6M/VOL_252D/PE)"
            ),
            "dropped_snapshot_features": [{"name": n, "family": f, "const_frac": c} for n, f, c in dropped],
        },
        "labels": {
            "anchor": "first trading day >= July 1 of fiscal_year+1",
            "horizon_trading_days": HORIZON_TDAYS,
            "price_source": "Yahoo Finance v8 chart API daily adjclose (bench/fetch_prices.py)",
            "forward_return": {
                "construction": "adjclose[t+126]/adjclose[t] - 1",
                "observed": int(m_ret.sum()),
            },
            "forward_realized_vol": {
                "construction": "std(daily simple returns over (t, t+126]) * sqrt(252)",
                "observed": int(m_vol.sum()),
            },
            "drawdown_exceedance": {
                "construction": (
                    "continuous forward_max_drawdown stored; binarized at benchmark "
                    "time as (dd <= train-split median dd); train split only"
                ),
                "observed": int(m_dd.sum()),
            },
        },
        "split": {
            "kind": "temporal on fiscal year",
            "mtnn_train_years": list(range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1)),
            "mtnn_val_year": VAL_YEAR,
            "harness_train": "fy <= 2021",
            "test": f"fy >= {TEST_CUT}",
            "purge_note": (
                "latest train label window ends ~Jan 2023 (fy2021: anchor Jul 2022 "
                "+ 126 tdays); earliest test anchor is Jul 2023 - no overlap"
            ),
        },
        "leakage_rules": (
            "features as-of fiscal-year-end (verified); labels strictly after the "
            "July 1 fy+1 anchor; snapshot features dropped by const-across-years "
            "audit; preprocessing (impute/standardize) fit on harness-train rows "
            "only inside run_benchmark.py"
        ),
    }
    sheet_path = args.out.with_name("datasheet.json")
    sheet_path.write_text(json.dumps(sheet, indent=2) + "\n")
    print(f"rows={len(fy)} labeled: ret={m_ret.sum()} vol={m_vol.sum()} dd={m_dd.sum()}")
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"wrote {args.out} and {sheet_path}")


if __name__ == "__main__":
    main()
