#!/usr/bin/env python3
"""Every named executive officer, with title, from SEC bulk Form 345 data sets.

Solo personal project, no connection to employer, built with public/free-tier only

WHY NOT THE PROXY. parse_def14a_xbrl.py gets the CEO from `ecd:PeoName` at 94.4%, and that
is where the CEO stops being available. The Pay-versus-Performance tags cover the PEO and
the AVERAGE of the other NEOs — `ecd:NonPeoNeoName` appears in ZERO of the 992 cached
filings, so the CFO/CTO/COO simply are not tagged there. The
`ecd:NamedExecutiveOfficersFnTextBlock` looked promising at 93.1% coverage and turned out
to be a compensation footnote naming only the PEO again ("the compensation reported for
Mr. Cook"). Reading individual names out of the Summary Compensation Table means table
scraping, which this pipeline measured at ~11% success on a 37-filing sample.

WHY NOT FORM 4 ONE AT A TIME. The cached EDGAR submissions index does list every Form 4 —
716 for a single ticker — which across 497 companies is roughly 350,000 documents. Correct
data, absurd request count.

SO: THE BULK QUARTERLY DATA SETS. SEC publishes every Form 3/4/5 as TSV, one ZIP per
quarter, at ~13 MB each. Two members carry everything needed:

    REPORTINGOWNER.tsv   ACCESSION_NUMBER, RPTOWNERCIK, RPTOWNERNAME,
                         RPTOWNER_RELATIONSHIP, RPTOWNER_TITLE
    SUBMISSION.tsv       ACCESSION_NUMBER, FILING_DATE, PERIOD_OF_REPORT,
                         ISSUERCIK, ISSUERNAME, ISSUERTRADINGSYMBOL

Joined on ACCESSION_NUMBER, that is (company, date) -> (person, title). One request per
quarter instead of 350,000, and the titles are filer-declared strings in a structured
column rather than anything inferred from prose.

FREE AND KEYLESS. No API key, no account. SEC asks for a declared User-Agent with contact
info and caps request rate; this makes one request per quarter, sequentially.

TITLES ARE FILER-WRITTEN FREE TEXT, and this file does not pretend otherwise. "CFO",
"Chief Financial Officer", "EVP and Chief Financial Officer" and "Chief Financial Officer
and Treasurer" are four strings for one job. A normalised ROLE is derived alongside the raw
title by keyword, and the RAW string is always kept so the normalisation can be checked or
redone. Rows whose relationship is Director-only are excluded: a director is not an officer,
and including them would silently inflate every count here.

    python pipeline/fetch_insider_officers.py --quarters 2024q1
    python pipeline/fetch_insider_officers.py            # 2015q1..latest
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "assets" / "real_data.json"
OUT = ROOT / "pipeline" / "data" / "officers.json"
CACHE = ROOT / "pipeline" / "cache" / "sec_form345"

UA = "vector-unified research jcdavis131@gmail.com"
URL = "https://www.sec.gov/files/structureddata/data/" "insider-transactions-data-sets/{q}_form345.zip"

# Keyword -> normalised role. ORDER MATTERS: the first match wins, so the more specific
# strings are listed before the ones they contain. "Chief Financial Officer" must be tested
# before "Officer", and CEO before "Chief" alone.
# THE "PRINCIPAL ___ OFFICER" FORMS ARE THE STATUTORY ONES and were missing from the first
# version, which is the worst kind of gap here: Section 16 filers use the SEC's own wording,
# so "Principal Accounting Officer" x11 and "Principal Financial Officer" landed in UNMAPPED
# while the colloquial "CAO"/"CFO" mapped fine. Measured on 2024q1 before this was fixed.
ROLE_RULES = [
    ("CEO", r"\bchief executive\b|\bCEO\b|\bprincipal executive officer\b"),
    ("CFO", r"\bchief financial\b|\bCFO\b|\bprincipal financial officer\b|\btreasurer\b"),
    ("COO", r"\bchief operating\b|\bCOO\b"),
    ("CTO", r"\bchief technology\b|\bchief technical\b|\bCTO\b"),
    ("CIO", r"\bchief information\b|\bCIO\b"),
    ("CAO", r"\bchief accounting\b|\bCAO\b|\bprincipal accounting officer\b|\bcontroller\b"),
    ("CLO", r"\bchief legal\b|\bgeneral counsel\b|\bCLO\b|\bGC\b"),
    ("CMO", r"\bchief marketing\b|\bCMO\b"),
    ("CHRO", r"\bchief human\b|\bCHRO\b|\bchief people\b"),
    ("CSO", r"\bchief scientific\b|\bchief science\b|\bchief strategy\b|\bCSO\b"),
    ("CRO", r"\bchief risk\b|\bchief revenue\b|\bCRO\b"),
    ("CMEDO", r"\bchief medical\b"),
    ("CCO", r"\bchief compliance\b|\bchief commercial\b|\bCCO\b"),
    # Catch-all for the long tail of real C-suite roles the named rules do not enumerate:
    # Chief Product / Administrative / Supply Chain / Investment / Brand / Growth /
    # Innovation Officer all appeared in one quarter. Placed AFTER the specific rules so it
    # only takes what they left, and kept as its own label rather than folded into an
    # existing role, because "some chief officer" is a weaker claim than "the CFO".
    ("CXO_OTHER", r"\bchief\b.{0,40}?\bofficer\b"),
    # VICE-PRESIDENT RULES MUST PRECEDE PRESIDENT. `\bpresident\b` matches inside "Senior
    # Vice President", so with PRESIDENT ordered first every VP in the file was relabelled
    # a president — which is how PRESIDENT came out as the largest role in the first full
    # run, 10,362 against CEO's 7,217. Caught by spot-checking AAPL 2023 and seeing Deirdre
    # O'Brien, "Senior Vice President", labelled PRESIDENT. A role distribution where the
    # #1 title outnumbers CEO by 40% was the visible symptom and I nearly shipped past it.
    ("EVP", r"\bexecutive vice president\b|\bEVP\b|\bexec\.?\s*v\.?\s*p\.?"),
    ("SVP", r"\bsenior vice president\b|\bSVP\b|\bsr\.?\s*v\.?\s*p\.?"),
    ("VP", r"\bvice president\b|\bVP\b|\bvice pres\b"),
    # Belt and braces: even ordered last, refuse to call a vice-president a president.
    ("PRESIDENT", r"(?<!vice )(?<!vice-)\bpresident\b|(?<!v\.)\bpres\.\b"),
    ("CHAIR", r"\bchairman\b|\bchairwoman\b|\bchairperson\b|\bchair\b"),
    ("MD", r"\bmanaging director\b"),
    ("HEAD", r"\bhead of\b"),
]
COMPILED = [(role, re.compile(pat, re.I)) for role, pat in ROLE_RULES]


def normalise_role(title: str) -> str | None:
    """Filer free text -> a coarse role, or None when nothing matches.

    None is a real answer meaning "this title does not map to a role I recognise". It is
    kept as None rather than bucketed into OTHER so the unmatched share stays visible in
    the run summary; a silently-bucketed remainder is how a normalisation stops being
    checkable.
    """
    if not title:
        return None
    for role, rx in COMPILED:
        if rx.search(title):
            return role
    return None


def quarters(start_year: int, end_year: int) -> list[str]:
    return [f"{y}q{q}" for y in range(start_year, end_year + 1) for q in (1, 2, 3, 4)]


def fetch_quarter(q: str, retries: int = 3) -> bytes | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / f"{q}_form345.zip"
    if local.exists() and local.stat().st_size > 1000:
        return local.read_bytes()
    req = urllib.request.Request(URL.format(q=q), headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            raw = urllib.request.urlopen(req, timeout=180).read()
            local.write_bytes(raw)
            return raw
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # quarter not published yet
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def read_tsv(z: zipfile.ZipFile, member: str):
    with z.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        yield from csv.DictReader(text, delimiter="\t")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quarters", nargs="*", help="e.g. 2024q1 2024q2; default 2015q1..2025q4")
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not REAL.exists():
        print(f"missing {REAL}")
        return 2
    real = json.loads(REAL.read_text(encoding="utf-8"))
    universe = {p["ticker"] for p in real["points"]}
    years = {int(p["year"]) for p in real["points"]}
    print(f"equities universe: {len(universe)} tickers, years {min(years)}-{max(years)}")

    qs = args.quarters or quarters(args.start, args.end)
    # (ticker, year) -> {(name, title)} ; a set because an officer files many Form 4s
    roster: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    seen_q = missing_q = 0

    for q in qs:
        raw = fetch_quarter(q)
        if raw is None:
            missing_q += 1
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile:
            missing_q += 1
            continue
        names = {n.upper(): n for n in z.namelist()}
        if "SUBMISSION.TSV" not in names or "REPORTINGOWNER.TSV" not in names:
            missing_q += 1
            continue

        # accession -> (ticker, year), restricted to the equities universe up front so the
        # much larger owner table is only scanned for rows that can matter
        acc: dict[str, tuple[str, int]] = {}
        for row in read_tsv(z, names["SUBMISSION.TSV"]):
            tkr = (row.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
            if tkr not in universe:
                continue
            date = (row.get("PERIOD_OF_REPORT") or row.get("FILING_DATE") or "").strip()
            m = re.search(r"(\d{4})", date)
            if not m:
                continue
            acc[row["ACCESSION_NUMBER"]] = (tkr, int(m.group(1)))

        for row in read_tsv(z, names["REPORTINGOWNER.TSV"]):
            key = acc.get(row.get("ACCESSION_NUMBER", ""))
            if key is None:
                continue
            rel = row.get("RPTOWNER_RELATIONSHIP") or ""
            if "officer" not in rel.lower():
                continue  # directors and 10% owners are not officers
            nm = (row.get("RPTOWNERNAME") or "").strip()
            ttl = (row.get("RPTOWNER_TITLE") or "").strip()
            if nm:
                roster[key].add((nm, ttl))
        seen_q += 1
        print(f"  {q}: {len(acc):>6} submissions in universe   " f"running roster keys {len(roster)}")

    print(f"\nquarters read {seen_q}, unavailable {missing_q}")

    out: dict[str, list[dict]] = {}
    role_counts: Counter = Counter()
    untitled = 0
    for (tkr, yr), people in sorted(roster.items()):
        rows = []
        for nm, ttl in sorted(people):
            role = normalise_role(ttl)
            role_counts[role or "UNMAPPED"] += 1
            if not ttl:
                untitled += 1
            rows.append({"name": nm, "title": ttl, "role": role})
        out[f"{tkr}_{yr}"] = rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    all_people = {r["name"] for rows in out.values() for r in rows}
    covered_tickers = {k.rsplit("_", 1)[0] for k in out}
    print(f"\n  (ticker, year) keys : {len(out)}")
    print(f"  distinct officers   : {len(all_people)}")
    print(f"  tickers covered     : {len(covered_tickers)}/{len(universe)}")
    print(f"  officer-rows        : {sum(len(v) for v in out.values())}")
    print(f"  rows with NO title  : {untitled}")
    print("\n  normalised role distribution:")
    for role, n in role_counts.most_common(24):
        print(f"    {role:12} {n:>6}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
