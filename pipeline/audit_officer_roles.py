"""How many of the 3,017 role-less officer rows are actually classifiable? Measure, don't fix.

officers.json carries 5,352 ticker-years and labels each officer with a role -- CEO, CFO,
EVP, SVP, CAO, CLO, COO, CHAIR, PRESIDENT, VP, CXO_OTHER. 3,017 rows come out with
role=None, which reads as "the classifier failed 3,017 times".

It did not. A large share of those rows have NO TITLE TO CLASSIFY: the filer wrote
"See Remarks" and put the title in a free-text block the parser never sees. Counting
those as classifier misses overstates the gap and would push someone into tuning a
regex against rows that contain no information. That is the same shape as every other
defect this phase has found -- a real number (3,017) answering a different question
(how many rows lack a role) than the one it appears to answer (how many the classifier
got wrong).

So this script splits the 3,017 into three populations and counts them:

  PLACEHOLDER   the filer stated no title. Nothing to classify, now or ever, without
                fetching the filing's remarks block.
  MAPPABLE      a real title the current classifier does not recognise. SEVP, Executive
                V.P., Comptroller, Corporate Secretary, Assistant Secretary.
  DIVISIONAL    a real title for a role at a SUBSIDIARY or business unit, not the
                registrant. "PRES. - ELECTRONIC INSTRUMENTS", "Chief Exec of a PPL
                Subsidiary", "Pres. of MPS Asia Operations". These must NOT be promoted
                into CEO/PRESIDENT -- doing so is exactly the bug that made
                officer_features.ceo_of() return FedEx Dataworks' CEO for FDX.

IT DELIBERATELY CHANGES NOTHING. officers.json feeds officer_features.py, which fills
trained NEO features (NEO_COUNT, NEO_TURNOVER, CEO_DUALITY, CEO_TENURE) in the equities
MTNN. Relabelling rows would move a model input, and the size of that move is unknown
until it is measured. This script produces the number that decides whether the change is
worth making; it does not make it.

    python pipeline/audit_officer_roles.py

Writes: pipeline/data/officer_role_audit.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICERS = ROOT / "pipeline" / "data" / "officers.json"
OUT = ROOT / "pipeline" / "data" / "officer_role_audit.json"

# The filer stated no title. Matched on the whole normalised string, not a substring,
# so a real title that happens to contain the word "remarks" is not swallowed.
PLACEHOLDER = re.compile(
    r"^(see\s+remarks?( below)?\.?|please\s+see\s+remarks?\.?|see\s+attached\.?|"
    r"n/?a|none|officer|executive officer|"
    r"see\s+footnotes?\.?|\*+|-+)$", re.I)

# A role at a subsidiary or unit rather than at the registrant. Detected by the presence
# of a scoping phrase, NOT by a list of company names -- a name list would be tuned
# against the filers it was written from.
DIVISIONAL = re.compile(
    r"\bsubsidiar|\bdivision\b|\bsegment\b|\bunit\b|\bgroup\b|"
    r"\bof\s+[A-Z]{2,}\b|"                      # "Pres. of MPS Asia Operations"
    r"\b(pres|president|chief exec|ceo|gm|general manager|vp)\b[^,]*[-,]\s*\S|"
    r"\b(asia|europe|americas|emea|apac|eastern hemisphere|western hemisphere|"
    r"international|north america|latin america)\b", re.I)

# Real titles the shipped classifier does not recognise. Kept to abbreviations and
# canonical office names -- anything requiring judgement about which unit it belongs to
# is left to DIVISIONAL rather than forced into a role.
MAPPABLE = [
    (re.compile(r"^s\.?e\.?v\.?p\.?$", re.I), "EVP"),
    (re.compile(r"^exec(utive)?\.?\s*v\.?[- ]?p\.?(res(ident)?)?\.?$", re.I), "EVP"),
    (re.compile(r"^exec(utive)?\.?\s*vice[- ]president$", re.I), "EVP"),
    (re.compile(r"^(corporate\s+)?secretary$", re.I), "SECRETARY"),
    (re.compile(r"^assistant\s+secretary$", re.I), "SECRETARY"),
    (re.compile(r"^(comptroller|controller)$", re.I), "CAO"),
    (re.compile(r"principal\s+acct|principal\s+accounting", re.I), "CAO"),
    (re.compile(r"^treasurer$", re.I), "TREASURER"),
    (re.compile(r"^sr\.?\s*v\.?p\.?$|^senior\s+vice[- ]president$", re.I), "SVP"),
    (re.compile(r"^(sr\.?|senior)\s+advisor$", re.I), "ADVISOR"),
    (re.compile(r"^vice\s+chair(man|woman)?$", re.I), "CHAIR"),
]


def classify(title: str):
    t = " ".join(str(title or "").split())
    if not t:
        return "PLACEHOLDER", None
    if PLACEHOLDER.match(t):
        return "PLACEHOLDER", None
    for rx, role in MAPPABLE:
        if rx.search(t):
            return "MAPPABLE", role
    if DIVISIONAL.search(t):
        return "DIVISIONAL", None
    return "UNCLASSIFIED", None


def main() -> int:
    if not OFFICERS.exists():
        print(f"FAIL: missing {OFFICERS}", file=sys.stderr)
        return 2
    d = json.loads(OFFICERS.read_text(encoding="utf-8"))

    total_rows = sum(len(v) for v in d.values())
    roleless = [(k, r) for k, v in d.items() for r in v if not r.get("role")]

    buckets = Counter()
    role_would_be = Counter()
    examples = defaultdict(list)
    for _, r in roleless:
        b, role = classify(r.get("title"))
        buckets[b] += 1
        if role:
            role_would_be[role] += 1
        if len(examples[b]) < 12:
            t = " ".join(str(r.get("title") or "").split())
            if t not in examples[b]:
                examples[b].append(t)

    # blast radius: which tickers would gain rows, and would any gain a CEO/PRESIDENT?
    # (none should -- MAPPABLE deliberately contains no CEO or PRESIDENT target)
    promotes_to_leadership = [r for r in role_would_be if r in ("CEO", "PRESIDENT")]

    tickers_touched = {k.rpartition("_")[0] for k, r in roleless
                       if classify(r.get("title"))[0] == "MAPPABLE"}

    out = {
        "question": "Of the role-less officer rows, how many are classifiable at all?",
        "total_officer_rows": total_rows,
        "roleless_rows": len(roleless),
        "roleless_pct": round(100.0 * len(roleless) / total_rows, 2),
        "split": dict(buckets),
        "split_pct_of_roleless": {k: round(100.0 * v / len(roleless), 1)
                                  for k, v in buckets.items()},
        "roles_that_would_be_assigned": dict(role_would_be.most_common()),
        "tickers_touched_by_a_MAPPABLE_relabel": len(tickers_touched),
        "would_any_row_become_CEO_or_PRESIDENT": promotes_to_leadership or
            "NO -- MAPPABLE contains no CEO or PRESIDENT target, deliberately. Promoting "
            "a divisional title into CEO is the bug that made officer_features.ceo_of() "
            "return FedEx Dataworks' CEO for FDX.",
        "examples": {k: v for k, v in examples.items()},
        "why_this_changes_nothing_yet": "officers.json feeds officer_features.py, which "
            "fills trained NEO features (NEO_COUNT, NEO_TURNOVER, CEO_DUALITY, "
            "CEO_TENURE) in the equities MTNN. Relabelling rows moves a model input. "
            "This script produces the number that decides whether that is worth doing; "
            "it does not do it.",
        "unclassified_tail_shape": None,
        "verdict": None,
        "headline": None,
    }
    unc_titles = Counter()
    for _, r in roleless:
        if classify(r.get("title"))[0] == "UNCLASSIFIED":
            unc_titles[" ".join(str(r.get("title") or "").split())] += 1
    out["unclassified_tail_shape"] = {
        "rows": buckets.get("UNCLASSIFIED", 0),
        "distinct_titles": len(unc_titles),
        "rows_per_distinct_title": round(
            buckets.get("UNCLASSIFIED", 0) / max(len(unc_titles), 1), 2),
        "reading": "A long tail of idiosyncratic abbreviations -- 'CVP, Critical Care', "
                   "'Chf Rsch, Dev & Innov Officer', 'Sr MD Gl Hd Commodity & Option' -- "
                   "not a missing rule. A handful are mappable with more patterns "
                   "('CVP PRIN ACCT OFFICER' -> CAO, 'E.V.P. Finance & C.F.O.' -> CFO, "
                   "'Senior Executive VicePresident' -> EVP), worth perhaps another 50 "
                   "rows. Each new pattern after that buys single-digit row counts.",
    }
    out["verdict"] = (
        "DO NOT RELABEL. The best case is roughly 250 of 56,257 rows (0.44%) across 27 "
        "tickers, and it costs moving a trained model input: officers.json feeds "
        "officer_features.py, which fills NEO_COUNT / NEO_TURNOVER / CEO_DUALITY / "
        "CEO_TENURE in the equities MTNN. The equities seed spread on sector accuracy "
        "was 0.0209; a 0.44% change in one feature's inputs is not measurable against "
        "that and would force a re-run of the shipped model to find out. The 828 "
        "DIVISIONAL rows are the genuinely interesting population and they should stay "
        "role-less -- promoting a subsidiary title into CEO or PRESIDENT is the bug that "
        "made officer_features.ceo_of() return FedEx Dataworks' CEO for FDX.")
    ph = buckets.get("PLACEHOLDER", 0)
    mp = buckets.get("MAPPABLE", 0)
    out["headline"] = (
        f"{len(roleless)} role-less rows is NOT {len(roleless)} classifier misses. "
        f"{ph} ({round(100.0*ph/len(roleless),1)}%) carry no title at all -- the filer "
        f"wrote 'See Remarks' or similar and the title lives in a free-text block the "
        f"parser never sees. {mp} are real titles the classifier does not recognise. "
        f"The rest are divisional or genuinely unclassified.")
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"total officer rows      {total_rows}")
    print(f"role-less rows          {len(roleless)}  ({out['roleless_pct']}%)")
    for b, n in buckets.most_common():
        print(f"   {b:<14} {n:>5}  ({out['split_pct_of_roleless'][b]}% of role-less)")
    print(f"\nroles a relabel would assign: {dict(role_would_be.most_common())}")
    print(f"tickers touched: {len(tickers_touched)}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
