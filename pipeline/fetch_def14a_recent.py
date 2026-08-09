"""Fetch each ticker's 2 most recent DEF14A filings (covers the 2 most recent
fiscal years, where the Pay-vs-Performance inline-XBRL rule -- effective for
proxies covering FY2022+ -- gives reliable machine-readable CEO/NEO comp).
Narrower than fetch_def14a_full.py (which grabs up to 11 OLDEST filings per
ticker first, backwards for this use case): here we want the newest, not the
oldest, and only 1-2 files/ticker instead of up to 11, so the full 500-ticker
universe is a few hundred fetches, not several thousand.

Run:  python pipeline/fetch_def14a_recent.py --limit 500 --start 0 --n 2
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import urllib.request

from fetch_def14a_full import collect_def14a_entries
from fetch_sec import USER_AGENT

CACHE_SUB = ROOT / "pipeline" / "cache" / "sec_submissions"
CACHE_DEF = ROOT / "pipeline" / "cache" / "sec_def14a"
CACHE_DEF.mkdir(parents=True, exist_ok=True)


def robust_fetch_html(url, out_path):
    if out_path.exists() and out_path.stat().st_size > 8000:
        return True
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
                if len(data) < 5000:
                    time.sleep(0.5)
                    continue
                out_path.write_bytes(data)
                print(f"  OK {out_path.name} {len(data)}")
                time.sleep(0.25)
                return True
        except Exception as e:
            print(f"  fetch fail attempt {attempt} {e} {url}")
            try:
                tmp = str(ROOT / "pipeline" / "cache" / "_def14a_tmp.html")
                cmd = [
                    "curl",
                    "-sL",
                    "--max-time",
                    "90",
                    "-H",
                    f"User-Agent: {USER_AGENT}",
                    "-o",
                    tmp,
                    url,
                ]
                subprocess.run(cmd, timeout=95)
                if Path(tmp).exists() and Path(tmp).stat().st_size > 5000:
                    out_path.write_bytes(Path(tmp).open("rb").read())
                    print(f"  curl OK {out_path.name} {out_path.stat().st_size}")
                    time.sleep(0.25)
                    return True
            except Exception:
                pass
            time.sleep(1 + attempt)
    return False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=2, help="most recent filings per ticker")
    args = ap.parse_args()
    uni = json.loads((ROOT / "pipeline" / "data" / "universe.json").read_text())
    subset = uni[args.start : args.start + args.limit]
    total = 0
    for i, entry in enumerate(subset):
        cik_pad = str(entry["cik"]).zfill(10)
        cik_nopad = str(int(entry["cik"]))
        ticker = entry["ticker"]
        entries = collect_def14a_entries(cik_pad)  # sorted ascending by date
        recent = entries[-args.n :]
        if not recent:
            continue
        c = 0
        for fdate, acc, primary in recent:
            if not acc or not primary:
                continue
            acc_nodash = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_nopad}/{acc_nodash}/{primary}"
            out = CACHE_DEF / f"{ticker}_{fdate}_{acc}.html"
            if robust_fetch_html(url, out):
                c += 1
                total += 1
        print(f"[{i + 1}/{len(subset)}] {ticker}: fetched {c}/{len(recent)}")
    print(f"\nTotal fetched: {total} HTML in {CACHE_DEF}")
