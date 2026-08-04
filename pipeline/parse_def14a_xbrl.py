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

# ---- EXTENSION, chosen by MEASURED YIELD rather than by which features were designed ----
#
# 33 of 118 columns in train_matrix.npz have EXACTLY zero observations, 12 of them the
# management_neo family (CEO_AGE, CEO_TENURE, CEO_FOUNDER_FLAG, CEO_PAY_RATIO,
# BOARD_INDEP_PCT, ...). The obvious move is to go extract those. THAT IS THE WRONG MOVE,
# and this file's own history says why: the table-scraping parser it replaced hit ~11%
# success on a 37-filing sample. None of those 12 fields is inline-XBRL tagged, so reviving
# them means going back to 11% guesswork about table structure.
#
# So the tags were COUNTED across all 992 cached filings before anything was written:
#
#     ecd:PeoName                          913   92.0%
#     ecd:PeoActuallyPaidCompAmt           960   96.8%
#     ecd:NonPeoNeoAvgCompActuallyPaidAmt  960   96.8%
#     ecd:TotalShareholderRtnAmt           960   96.8%
#     ecd:PeerGroupTotalShareholderRtnAmt  958   96.6%
#     ecd:NonPeoNeoName                      0    0.0%   <- so NEO_COUNT stays unavailable
#
# A legally-required machine-readable fact at 92-97% beats a designed-but-unfillable field
# at 0% or a scraped guess at 11%. NEO_COUNT is explicitly NOT added: the tag that would
# supply it appears in zero filings, and inferring it from a text block would be exactly
# the guesswork this parser exists to avoid.
#
# ecd:PeoName is the one that matters most beyond compensation: it is the CEO's NAME, which
# is an ENTITY, not a statistic. It is what lets a company row join to a person.
# MATCH THE ELEMENT, NOT THE TEXT AFTER THE ATTRIBUTE. A `>([^<]*)<` capture assumes the
# value is bare text, and inline XBRL routinely splits a name across nested inline tags:
#
#     <ix:nonNumeric name="ecd:PeoName" ...>J<div ...>uan Luciano</div></ix:nonNumeric>
#     <ix:nonNumeric ... name="ecd:PeoName"><span ...>Pat Gallagher</span></ix:nonNumeric>
#
# The bare-text form found 834 of the 913 filings carrying the tag; the missing 79 all look
# like the above. Matching the whole ix:nonNumeric element and stripping markup afterwards
# reaches them, and also tolerates `name=` appearing anywhere in the attribute list.
PEO_NAME_RE = re.compile(
    r'<ix:nonNumeric[^>]*name="ecd:PeoName"[^>]*>(.*?)</ix:nonNumeric>',
    re.DOTALL | re.IGNORECASE)
PEO_PAID_RE = re.compile(r'name="ecd:PeoActuallyPaidCompAmt"[^>]*>([^<]*)<')
NEO_PAID_RE = re.compile(r'name="ecd:NonPeoNeoAvgCompActuallyPaidAmt"[^>]*>([^<]*)<')
TSR_RE = re.compile(r'name="ecd:TotalShareholderRtnAmt"[^>]*>([^<]*)<')
PEER_TSR_RE = re.compile(r'name="ecd:PeerGroupTotalShareholderRtnAmt"[^>]*>([^<]*)<')

# Inline XBRL wraps values in nested spans and escapes entities; a name grabbed raw can
# arrive as "John&#160;Smith" or with stray markup. Normalised here, once.
_WS = re.compile(r"[\s ]+")
_TAGS = re.compile(r"<[^>]+>")


def _clean_name(s: str) -> str | None:
    # Tags are removed WITHOUT inserting a space. Inline XBRL splits names mid-word
    # ("J<div>uan Luciano</div>"), so substituting a space yields "J uan Luciano". Deleting
    # the markup yields "Juan Luciano", which is the actual filed name.
    s = _TAGS.sub("", s)
    s = s.replace("&#160;", " ").replace("&nbsp;", " ").replace("&amp;", "&")
    s = _WS.sub(" ", s).strip(" ,;.")
    # A REQUIRE-TWO-TOKENS RULE WAS TOO STRICT AND THREW AWAY REAL DATA. Sampling the 32
    # rejects showed 30 were surnames the filer tagged alone — Ford, Tomczyk, Rochow,
    # Lance, Kumar, Novakovic, Hazen, Demetriou — and only 2 were junk, the footnote
    # markers "(1)" and "(2)". Rejecting 30 real executives to catch 2 markers is the wrong
    # trade, so the rule now only demands that the value contain a letter and be plausibly
    # name-length. It does NOT try to decide whether a string is a real person; this file
    # cannot know that, and pretending otherwise is how the strict rule got written.
    if not (2 <= len(s) <= 120):
        return None
    return s if any(c.isalpha() for c in s) else None


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

    # Same first-occurrence rule as the two original tags: PVP tables render newest-first,
    # so index 0 is the row for the fiscal year this filing is being assigned to. Applying a
    # different rule to the new tags would silently mix years within one record.
    name_m = PEO_NAME_RE.findall(html)
    peo_paid = PEO_PAID_RE.findall(html)
    neo_paid = NEO_PAID_RE.findall(html)
    tsr = TSR_RE.findall(html)
    peer = PEER_TSR_RE.findall(html)

    ceo_name = _clean_name(name_m[0]) if name_m else None
    tsr_v = _num(tsr[0]) if tsr else None
    peer_v = _num(peer[0]) if peer else None
    return {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "filing_date": fdate,
        "file": html_path.name,
        "CEO_TOTAL_COMP": ceo_comp,
        "AVG_NEO_COMP": avg_neo_comp,
        "CEO_NAME": ceo_name,
        "CEO_COMP_ACTUALLY_PAID": _num(peo_paid[0]) if peo_paid else None,
        "AVG_NEO_COMP_ACTUALLY_PAID": _num(neo_paid[0]) if neo_paid else None,
        "TSR_INDEXED": tsr_v,
        "PEER_TSR_INDEXED": peer_v,
        # Company TSR minus its OWN stated peer group, both indexed to $100 by SEC rule.
        # This is the filer's declared peer set, not a sector bucket I chose, which makes it
        # a better relative-return measure than SECTOR_REL_RET_12M would have been.
        "TSR_VS_PEER": (round(tsr_v - peer_v, 4)
                        if tsr_v is not None and peer_v is not None else None),
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

    # PER-FIELD YIELD, printed every run. A parser that reports only a record count hides
    # the case that matters: records written with the new fields all None, which looks like
    # success and delivers nothing. This is the same defect the estate keeps finding — a
    # real number answering a different question than the one it appears to answer.
    fields = ("CEO_TOTAL_COMP", "AVG_NEO_COMP", "CEO_NAME", "CEO_COMP_ACTUALLY_PAID",
              "AVG_NEO_COMP_ACTUALLY_PAID", "TSR_INDEXED", "PEER_TSR_INDEXED",
              "TSR_VS_PEER")
    n = max(1, len(by_key))
    print("\n  per-field yield over the written records:")
    for f_ in fields:
        c = sum(1 for r in by_key.values() if r.get(f_) is not None)
        print(f"    {f_:28} {c:>5}/{n}  {100*c/n:>5.1f}%")
    names = {r["CEO_NAME"] for r in by_key.values() if r.get("CEO_NAME")}
    print(f"\n  distinct CEO names (the executive ENTITY): {len(names)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(by_key, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
