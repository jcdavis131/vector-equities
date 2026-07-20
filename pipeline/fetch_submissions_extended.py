"""Fetch submissions plus additional files for full 2015+ history"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from fetch_sec import robust_fetch_json

CACHE = ROOT / "pipeline" / "cache" / "sec_submissions"
CACHE.mkdir(parents=True, exist_ok=True)


def fetch_json_file(cik_pad, file_name):
    out = CACHE / f"{cik_pad}_{file_name}"
    # also try cached
    if out.exists() and out.stat().st_size > 5000:
        try:
            return json.loads(out.read_text())
        except:
            pass
    url = (
        f"https://data.sec.gov/submissions/{cik_pad}_{file_name}"
        if "_" in file_name
        else f"https://data.sec.gov/submissions/{file_name}"
    )
    # Actually format: https://data.sec.gov/submissions/CIK0000066740-submissions-001.json -> need full name, url base https://data.sec.gov/submissions/
    if not url.startswith("https://data.sec.gov/submissions/CIK"):
        url = f"https://data.sec.gov/submissions/{file_name}"
    # The correct URL is https://data.sec.gov/submissions/CIK{pad}-submissions-001.json? No, API says /submissions/{file}
    # filings.files name already includes CIK prefix, so URL is https://data.sec.gov/submissions/{name}
    url = f"https://data.sec.gov/submissions/{file_name}"
    print(f"Fetching additional {url}")
    data = robust_fetch_json(url)
    if data:
        out.write_text(json.dumps(data))
        time.sleep(0.25)
    return data


def get_all_def14a_for_cik(cik_pad):
    sub_path = CACHE / f"sub_{cik_pad}.json"
    if not sub_path.exists():
        return []
    main = json.loads(sub_path.read_text())
    all_filings = []
    # recent
    recent = main.get("filings", {}).get("recent", {})
    all_filings.append(recent)
    # files
    files = main.get("filings", {}).get("files", [])
    for f in files:
        name = f["name"]
        # only fetch if filingFrom <= 2025 and filingTo >= 2015 potentially contains our range
        # Heuristic: if filingTo year <2015 skip, if filingFrom >2025 skip
        try:
            from_y = int(f["filingFrom"][:4])
            to_y = int(f["filingTo"][:4])
            if to_y < 2015 or from_y > 2025:
                continue
        except:
            pass
        extra_path = CACHE / f"{cik_pad}_{name}"
        if not extra_path.exists():
            data = fetch_json_file(cik_pad, name)
        else:
            try:
                data = json.loads(extra_path.read_text())
            except:
                data = None
        if data:
            all_filings.append(data)
    # Now collect DEF14A
    results = []
    for chunk in all_filings:
        forms = chunk.get("form", [])
        dates = chunk.get("filingDate", [])
        acc = chunk.get("accessionNumber", [])
        docs = chunk.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if "DEF 14A" in form:
                try:
                    yr = int(dates[i][:4]) if i < len(dates) else 0
                    if yr < 2015 or yr > 2025:
                        continue
                except:
                    continue
                results.append(
                    (
                        dates[i] if i < len(dates) else "",
                        acc[i] if i < len(acc) else "",
                        docs[i] if i < len(docs) else "",
                    )
                )
    results = sorted(set(results))
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    uni = json.loads((ROOT / "pipeline" / "data" / "universe.json").read_text())
    subset = uni[args.start : args.start + args.limit]
    for entry in subset:
        cik = str(entry["cik"]).zfill(10)
        ticker = entry["ticker"]
        def14as = get_all_def14a_for_cik(cik)
        print(f"{ticker} {cik} DEF14A total 2015-2025: {len(def14as)} {def14as[:3]}")
