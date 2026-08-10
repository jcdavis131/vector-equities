"""Fetch daily adjusted price history for the benchmark panel tickers.

Source: Yahoo Finance v8 chart API — the same market-data source the repo's
pipeline documents and uses (pipeline/fetch_market_history.py via yfinance).
Plain HTTPS GETs via curl; one JSON per ticker written to --out-dir as
{ticker, dates, close, adjclose, volume}. Already-fetched tickers are skipped,
so the fetch is resumable. Tickers Yahoo no longer serves (delisted/renamed)
are recorded in <out-dir>/../prices_missing.json and their panel rows are
excluded per target by label masks downstream (never imputed).

Usage:
    python bench/fetch_prices.py --out-dir /path/to/prices
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MATRIX_BLOB = "02d107b4f01cb6ecfab8ee0b162d440f42638eae"

PERIOD1 = 1356998400  # 2013-01-01 UTC (lookback margin for feature audits)
PERIOD2 = 1786665600  # 2026-08-14 UTC
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


def panel_tickers() -> list[str]:
    raw = subprocess.run(["git", "cat-file", "blob", MATRIX_BLOB], cwd=REPO, capture_output=True, check=True).stdout
    tmp = Path("/tmp/_equities_panel.npz")
    tmp.write_bytes(raw)
    z = np.load(tmp, allow_pickle=True)
    return sorted(set(z["ticker"].astype(str)))


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


def fetch_one(t: str) -> dict | None:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(t)}"
        f"?period1={PERIOD1}&period2={PERIOD2}&interval=1d&events=div%2Csplit"
    )
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["curl", "-sS", "--max-time", "45", "-H", f"User-Agent: {UA}", url],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr[:200])
            j = json.loads(r.stdout)
            res = j.get("chart", {}).get("result")
            if not res:
                err = j.get("chart", {}).get("error")
                if err and err.get("code") in ("Not Found", "not-found"):
                    return None
                raise RuntimeError(str(err)[:200])
            res = res[0]
            ts = res.get("timestamp") or []
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            if not ts or adj is None:
                return None
            return {
                "ticker": t,
                "dates": [time.strftime("%Y-%m-%d", time.gmtime(x)) for x in ts],
                "close": q.get("close"),
                "adjclose": adj,
                "volume": q.get("volume"),
            }
        except Exception as e:
            if attempt == 2:
                print(f"FAIL {t}: {e}", flush=True)
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tickers = panel_tickers()
    ok = 0
    missing: list[str] = []
    for i, t in enumerate(tickers):
        fp = args.out_dir / f"{t}.json"
        if fp.exists():
            ok += 1
            continue
        data = fetch_one(t)
        if data is None:
            missing.append(t)
        else:
            fp.write_text(json.dumps(data))
            ok += 1
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(tickers)}] ok={ok} missing={len(missing)}", flush=True)
        time.sleep(args.sleep)
    (args.out_dir.parent / "prices_missing.json").write_text(json.dumps(missing, indent=1))
    print(f"DONE ok={ok} missing={len(missing)}: {missing}", flush=True)


if __name__ == "__main__":
    main()
