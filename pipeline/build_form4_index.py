#!/usr/bin/env python3
"""Build form4_index_all.jsonl from the cached SEC Form 3/4/5 quarterly datasets.

WHY
---
`build_real_v5_career.py:load_form4()` reads `pipeline/data/form4_index_all.jsonl`
and prints "No Form4 index found" when it is absent -- which it always has been.
INSIDER_NET_12M therefore fell back to a hardcoded constant for every row.

Meanwhile 470 MB of the real source has been sitting in
`pipeline/cache/sec_form345/` -- 46 quarterly ZIPs, 2015q1 onward, i.e. the
FULL fiscal-year range the matrix covers (unlike 13F, which starts 2024).

WHAT IT COMPUTES
----------------
One record per Form 3/4/5 submission, joining two members of each quarterly ZIP:

  SUBMISSION.tsv        ACCESSION_NUMBER -> ISSUERCIK, ISSUERTRADINGSYMBOL,
                        FILING_DATE, PERIOD_OF_REPORT, DOCUMENT_TYPE
  NONDERIV_TRANS.tsv    per accession: TRANS_CODE, TRANS_SHARES,
                        TRANS_PRICEPERSHARE, TRANS_ACQUIRED_DISP_CD

`net_shares` sums TRANS_SHARES signed by TRANS_ACQUIRED_DISP_CD (A = acquired,
positive; D = disposed, negative). `net_value` does the same weighted by price
where a price is reported.

Only OPEN-MARKET codes count toward the net by default (`P` purchase, `S`
sale). Codes like `A` (grant), `M` (option exercise), `F` (tax withholding) are
compensation mechanics rather than a director expressing a view, and folding
them in makes the signal mostly payroll. Both are emitted -- `net_shares` for
open-market only, `net_shares_all` for every code -- so a consumer can choose,
and neither is silently assumed.

NOTHING IS DEFAULTED. A submission with no parseable transactions gets
net_shares = null, not 0. Zero means "transactions netted to zero", null means
"no usable transaction rows" -- they are different facts and the matrix builder
treats a null as mask=0.

Usage:
    python pipeline/build_form4_index.py                  # all quarters
    python pipeline/build_form4_index.py --limit 2        # smoke test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "sec_form345"
OUT = ROOT / "pipeline" / "data" / "form4_index_all.jsonl"

# Open-market transactions: the insider chose to buy or sell.
OPEN_MARKET = {"P", "S"}

# NONDERIV_TRANS carries every security class a filer reports -- common,
# preferred, units, warrants-as-nonderivative, and assorted one-off instruments
# ("NOTIS GLC"). Summing share COUNTS across classes is an apples-to-oranges
# error: 100,000,000 preferred at $100 is not comparable to common at $15.
# Restrict to common-equivalent titles, which is what an insider-conviction
# signal is about. Verified against a quarter's title distribution: "common
# stock", "class a common stock", "common shares", "ordinary shares" and
# "class c capital stock" cover the overwhelming majority.
COMMON_HINTS = ("common", "ordinary", "capital stock")

# Form 4 filers declare their relationship to the issuer in REPORTINGOWNER.tsv:
# Director, Officer, TenPercentOwner, Other. A feature called INSIDER_NET_12M
# should mean insiders, so "Other" alone does not qualify.
#
# This is not an outlier filter -- it is the feature's own definition. It does
# happen to exclude erroneous filings: BAC FY2016 carried a single submission
# from "YNOFACE Holdings Inc" (relationship: Other) reporting 4.2 BILLION common
# shares, roughly 40% of Bank of America. EDGAR accepts Form 4 filings without
# verifying them, so the record genuinely contains such entries; they are real
# SEC data and obviously not real insider activity. Excluding them by WHO FILED
# is defensible in a way that clipping by magnitude would not be.
INSIDER_RELATIONSHIPS = ("director", "officer", "tenpercentowner")


def is_insider(rel: str) -> bool:
    r = (rel or "").strip().lower()
    return any(k in r for k in INSIDER_RELATIONSHIPS)


def is_common(title: str) -> bool:
    t = (title or "").strip().lower()
    return any(h in t for h in COMMON_HINTS)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_sec_date(s: str) -> str | None:
    """SEC ships '31-MAR-2015'. Return ISO 'YYYY-MM-DD', or None if unparseable."""
    s = (s or "").strip()
    if not s:
        return None
    parts = s.split("-")
    if len(parts) != 3:
        return None
    dd, mon, yyyy = parts
    m = MONTHS.get(mon.upper()[:3])
    if not m or not yyyy.isdigit() or not dd.isdigit():
        return None
    return f"{int(yyyy):04d}-{m:02d}-{int(dd):02d}"


def _num(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_tsv(zf: zipfile.ZipFile, member: str):
    if member not in zf.namelist():
        return
    with zf.open(member) as fh:
        r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
        yield from r


def process_zip(path: Path) -> list[dict]:
    out = []
    with zipfile.ZipFile(path) as zf:
        subs = {}
        for row in read_tsv(zf, "SUBMISSION.tsv"):
            acc = (row.get("ACCESSION_NUMBER") or "").strip()
            if not acc:
                continue
            subs[acc] = {
                "accession": acc,
                "ticker": (row.get("ISSUERTRADINGSYMBOL") or "").strip().upper() or None,
                "issuer_cik": (row.get("ISSUERCIK") or "").strip() or None,
                "issuer": (row.get("ISSUERNAME") or "").strip() or None,
                "filing_date": parse_sec_date(row.get("FILING_DATE", "")),
                "period": parse_sec_date(row.get("PERIOD_OF_REPORT", "")),
                "doc_type": (row.get("DOCUMENT_TYPE") or "").strip() or None,
            }

        insiders = set()
        for row in read_tsv(zf, "REPORTINGOWNER.tsv"):
            acc = (row.get("ACCESSION_NUMBER") or "").strip()
            if acc and is_insider(row.get("RPTOWNER_RELATIONSHIP", "")):
                insiders.add(acc)

        agg = defaultdict(lambda: {"n": 0, "open_sh": 0.0, "all_sh": 0.0,
                                   "open_val": 0.0, "any_open": False, "any_all": False})
        for row in read_tsv(zf, "NONDERIV_TRANS.tsv"):
            acc = (row.get("ACCESSION_NUMBER") or "").strip()
            if not acc:
                continue
            if acc not in insiders:
                continue
            shares = _num(row.get("TRANS_SHARES"))
            if shares is None:
                continue
            if not is_common(row.get("SECURITY_TITLE", "")):
                continue
            code = (row.get("TRANS_CODE") or "").strip().upper()
            ad = (row.get("TRANS_ACQUIRED_DISP_CD") or "").strip().upper()
            if ad not in ("A", "D"):
                continue
            sign = 1.0 if ad == "A" else -1.0
            price = _num(row.get("TRANS_PRICEPERSHARE"))
            a = agg[acc]
            a["n"] += 1
            a["all_sh"] += sign * shares
            a["any_all"] = True
            # An OPEN-MARKET trade has a price. A P/S row reporting
            # TRANS_PRICEPERSHARE 0.00 is a transfer, a gift, an error or a
            # hoax -- not someone buying or selling at market. Requiring a
            # positive price is the definition of the thing this column
            # measures, not an outlier filter.
            #
            # It is what excludes GOOGL FY2017: "Lee Antonio", filing as
            # TenPercentOwnerOther, reported a PURCHASE of 4,500,000,000 common
            # shares at $0.00 -- roughly six times Alphabet's actual Class A
            # count. EDGAR does not verify Form 4 submissions, so entries like
            # this are genuinely in the record.
            if code in OPEN_MARKET and price is not None and price > 0:
                a["open_sh"] += sign * shares
                a["any_open"] = True
                a["open_val"] += sign * shares * price

        for acc, sub in subs.items():
            if not sub["ticker"] or not sub["filing_date"]:
                continue
            a = agg.get(acc)
            rec = dict(sub)
            # null, not 0 -- "no usable transaction rows" is not "netted to zero"
            rec["n_trans"] = a["n"] if a else 0
            rec["net_shares"] = a["open_sh"] if (a and a["any_open"]) else None
            rec["net_shares_all"] = a["all_sh"] if (a and a["any_all"]) else None
            rec["net_value"] = a["open_val"] if (a and a["any_open"] and a["open_val"]) else None
            out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N quarters (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    zips = sorted(CACHE.glob("*_form345.zip"))
    if not zips:
        raise SystemExit(f"no quarterly ZIPs under {CACHE}")
    if a.limit:
        zips = zips[: a.limit]
    print(f"{len(zips)} quarterly archive(s)")

    total = 0
    tickers: set[str] = set()
    with_net = 0
    with Path(a.out).open("w", encoding="utf-8") as fh:
        for i, z in enumerate(zips, 1):
            try:
                recs = process_zip(z)
            except Exception as e:  # noqa: BLE001
                print(f"  {z.name}: FAILED ({type(e).__name__}: {e})")
                continue
            for r in recs:
                fh.write(json.dumps(r) + "\n")
                tickers.add(r["ticker"])
                if r["net_shares"] is not None:
                    with_net += 1
            total += len(recs)
            print(f"  [{i}/{len(zips)}] {z.name}: {len(recs)} submissions (running total {total})")

    print(f"\nWrote {total} submissions -> {a.out}")
    print(f"  distinct issuer tickers : {len(tickers)}")
    print(f"  with open-market net    : {with_net} ({with_net / max(total,1) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
