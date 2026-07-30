"""Extract real CEO/NEO total comp from DEF14A inline-XBRL Pay-vs-Performance
tags (SEC Item 402(v), effective for proxies covering FY2022+).

Far more reliable than table-scraping: `ecd:PeoTotalCompAmt` and
`ecd:NonPeoNeoAvgTotalCompAmt` are machine-readable facts the filer is
legally required to tag correctly, not a heuristic guess at table structure
(parse_def14a.py's regex table scraper hit ~11% success on a 37-filing
sample; this hits the tag directly).

Fiscal-year mapping is a documented approximation, not exact: takes the
FIRST (most-recent, tables render newest-first) occurrence of each tag and
assigns it to `filing_year - 1` (proxies are typically filed the spring
after fiscal year-end). This is wrong for the handful of tickers with a
non-calendar fiscal year (e.g. ACN's is Sept-Aug) -- same precision-level
tradeoff market_features.py already makes for "as of" dates, not a new
approximation. Pre-2023 filings have no PVP tags at all and are correctly
skipped, not guessed at.

Run:  python pipeline/parse_def14a_xbrl.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DEF = ROOT / "pipeline" / "cache" / "sec_def14a"
OUT = ROOT / "pipeline" / "data" / "def14a_comp.json"

PEO_RE = re.compile(r'name="ecd:PeoTotalCompAmt"[^>]*>([^<]*)<')
NEO_AVG_RE = re.compile(r'name="ecd:NonPeoNeoAvgTotalCompAmt"[^>]*>([^<]*)<')


def _num(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "")
    if not s or s in ("-", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_one(html_path: Path) -> dict | None:
    stem_parts = html_path.stem.split("_")
    if len(stem_parts) < 2:
        return None
    ticker = stem_parts[0]
    fdate = stem_parts[1]
    try:
        filing_year = int(fdate[:4])
    except ValueError:
        return None
    fiscal_year = filing_year - 1

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    peo_m = PEO_RE.findall(html)
    neo_m = NEO_AVG_RE.findall(html)
    ceo_comp = _num(peo_m[0]) if peo_m else None
    avg_neo_comp = _num(neo_m[0]) if neo_m else None
    if ceo_comp is None and avg_neo_comp is None:
        return None
    return {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "filing_date": fdate,
        "file": html_path.name,
        "CEO_TOTAL_COMP": ceo_comp,
        "AVG_NEO_COMP": avg_neo_comp,
    }


def main() -> None:
    files = sorted(CACHE_DEF.glob("*.html"))
    print(f"scanning {len(files)} cached DEF14A filings")
    by_key: dict[str, dict] = {}
    hits = 0
    for f in files:
        row = parse_one(f)
        if row is None:
            continue
        key = f"{row['ticker']}_{row['fiscal_year']}"
        # a ticker can have 2 filings covering the same inferred fiscal year
        # if filed close together across a calendar boundary -- keep the one
        # from the later filing (more likely the primary/correct year match).
        prev = by_key.get(key)
        if prev is None or row["filing_date"] > prev["filing_date"]:
            by_key[key] = row
        hits += 1
    print(f"{hits}/{len(files)} filings had at least one PVP tag")
    print(f"{len(by_key)} unique ticker/fiscal-year records")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(by_key, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
