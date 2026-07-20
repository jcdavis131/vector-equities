"""Form 4 index for top N — builds chronological insider timeline"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

CACHE_SUB = ROOT / "pipeline" / "cache" / "sec_submissions"
CACHE_F4 = ROOT / "pipeline" / "cache" / "sec_form4"
CACHE_F4.mkdir(parents=True, exist_ok=True)


def get_all_form4(cik_pad, since_year=2015, max_per_ticker=200):
    main_path = CACHE_SUB / f"sub_{cik_pad}.json"
    if not main_path.exists():
        return []
    main = json.loads(main_path.read_text())
    chunks = [main.get("filings", {}).get("recent", {})]
    for f in main.get("filings", {}).get("files", []):
        try:
            int(f.get("filingFrom", "2000")[:4])
            to_y = int(f.get("filingTo", "2025")[:4])
            if to_y < since_year:
                continue
        except:
            pass
        extra = CACHE_SUB / f"{cik_pad}_{f['name']}"
        if extra.exists():
            try:
                chunks.append(json.loads(extra.read_text()))
            except:
                pass
    entries = []
    for chunk in chunks:
        forms = chunk.get("form", [])
        dates = chunk.get("filingDate", [])
        acc = chunk.get("accessionNumber", [])
        docs = chunk.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if form != "4":
                continue
            try:
                d = dates[i]
                y = int(d[:4])
                if y < since_year:
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
    # dedupe by accession, sort newest first
    uniq = {}
    for d, a, doc in entries:
        uniq[a] = (d, a, doc)
    sorted_entries = sorted(uniq.values(), reverse=True)
    return sorted_entries[:max_per_ticker]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--since", type=int, default=2020)
    args = ap.parse_args()
    uni = json.loads((ROOT / "pipeline" / "data" / "universe.json").read_text())
    subset = uni[args.start : args.start + args.limit]
    master = []
    for entry in subset:
        cik_pad = str(entry["cik"]).zfill(10)
        ticker = entry["ticker"]
        f4 = get_all_form4(cik_pad, since_year=args.since, max_per_ticker=300)
        print(
            f"{ticker} {cik_pad} Form4 since {args.since}: {len(f4)} example {f4[:2]}"
        )
        for d, a, doc in f4:
            master.append(
                {
                    "ticker": ticker,
                    "cik": cik_pad,
                    "filing_date": d,
                    "accession": a,
                    "primaryDoc": doc,
                }
            )
    out = Path("pipeline/data/form4_index.jsonl")
    with open(out, "w") as f:
        for rec in master:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {out} {len(master)} Form4 index records")
