"""Fetch DEF14A HTML for top N using extended submissions (all years 2015-2025)"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import urllib.request

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
                    print(f"  small {len(data)} {url}")
                    time.sleep(0.5)
                    continue
                out_path.write_bytes(data)
                print(f"  OK {out_path.name} {len(data)}")
                time.sleep(0.25)
                return True
        except Exception as e:
            print(f"  fetch fail attempt {attempt} {e} {url}")
            # curl fallback
            try:
                tmp = "/tmp/def14a_tmp2.html"
                import subprocess

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
                if os.path.exists(tmp) and os.path.getsize(tmp) > 5000:
                    out_path.write_bytes(open(tmp, "rb").read())
                    print(f"  curl OK {out_path.name} {out_path.stat().st_size}")
                    time.sleep(0.25)
                    return True
            except:
                pass
            time.sleep(1 + attempt)
    return False


def collect_def14a_entries(cik_pad):
    """Collect all DEF14A entries from recent + additional files"""
    main_path = CACHE_SUB / f"sub_{cik_pad}.json"
    if not main_path.exists():
        return []
    main = json.loads(main_path.read_text())
    chunks = []
    chunks.append(main.get("filings", {}).get("recent", {}))
    files = main.get("filings", {}).get("files", [])
    for f in files:
        name = f["name"]
        try:
            from_y = int(f.get("filingFrom", "2000")[:4])
            to_y = int(f.get("filingTo", "2025")[:4])
            if to_y < 2015 or from_y > 2025:
                continue
        except:
            pass
        extra_path = CACHE_SUB / f"{cik_pad}_{name}"
        if extra_path.exists():
            try:
                chunks.append(json.loads(extra_path.read_text()))
            except:
                pass
    entries = []
    for chunk in chunks:
        forms = chunk.get("form", [])
        dates = chunk.get("filingDate", [])
        acc = chunk.get("accessionNumber", [])
        docs = chunk.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if "DEF 14A" not in form:
                continue
            try:
                d = dates[i] if i < len(dates) else ""
                y = int(d[:4]) if d else 0
                if y < 2015 or y > 2025:
                    continue
            except:
                continue
            entries.append(
                (
                    dates[i] if i < len(dates) else "",
                    acc[i] if i < len(acc) else "",
                    docs[i] if i < len(docs) else "",
                )
            )
    # dedupe by accession
    uniq = {}
    for d, acc, doc in entries:
        uniq[acc] = (d, acc, doc)
    return sorted(uniq.values())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    uni = json.loads((ROOT / "pipeline" / "data" / "universe.json").read_text())
    subset = uni[args.start : args.start + args.limit]
    total = 0
    for entry in subset:
        cik_pad = str(entry["cik"]).zfill(10)
        cik_nopad = str(int(entry["cik"]))
        ticker = entry["ticker"]
        entries = collect_def14a_entries(cik_pad)
        print(f"\n{ticker} {cik_pad} DEF14A 2015-2025 total {len(entries)}")
        c = 0
        for fdate, acc, primary in entries:
            if not acc or not primary:
                continue
            acc_nodash = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_nopad}/{acc_nodash}/{primary}"
            out = CACHE_DEF / f"{ticker}_{fdate}_{acc}.html"
            if robust_fetch_html(url, out):
                c += 1
                total += 1
            if c >= 11:
                break
        print(f"  fetched {c} for {ticker}")
    print(f"\nTotal fetched all: {total} HTML in {CACHE_DEF}")
