#!/usr/bin/env python3
"""Fill the management_neo columns that officers.json can actually support — and only those.

Solo personal project, no connection to employer, built with public/free-tier only

train_matrix.npz has 33 columns with EXACTLY zero observations, twelve of them the
management_neo family. pipeline/data/officers.json now carries 56,257 officer-rows over
10,247 distinct people, covering 4,550 of 4,831 company-years (94.2%) in every year from
2015 to 2024. FOUR of the twelve become computable. Eight do not, and this file fills four.

    NEO_COUNT       officers on file for that company-year          direct count
    NEO_TURNOVER    share of last year's officers no longer present set difference
    CEO_DUALITY     the CEO's own title also says chair/chairman    title string
    CEO_TENURE      years since this CEO first appears in the data  LEFT-CENSORED, see below

STILL DEAD AND HONESTLY SO, because Form 345 reporting-owner records do not contain them:
CEO_AGE, CEO_FOUNDER_FLAG, CEO_EQUITY_PCT, CEO_PAY_RATIO, BOARD_INDEP_PCT, BOARD_SIZE,
INSIDER_OWN_PCT, CEO_PAY_VS_SECTOR. Directors were excluded from the officer extract by
design, so the two BOARD_ fields are not merely missing here — this source cannot ever
supply them. Filling a column with a plausible guess is worse than leaving it masked: the
model would train on it and every downstream reader would treat it as measured.

CEO_TENURE IS LEFT-CENSORED AND THE MASK SAYS SO. The panel starts in 2015. A CEO appointed
in 2004 first APPEARS in 2015, so a naive "years since first seen" reports tenure 0 for a
veteran — a real number answering a different question than the one it appears to answer.
Tenure is therefore written ONLY when the first appearance is after the company's own first
year in the panel, i.e. when an actual transition was observed. Everyone else is masked
unobserved rather than given a number that would read as measured.

NEO_TURNOVER NEEDS A PRIOR YEAR, so a company's first panel year is masked too. Reporting
turnover 0.0 there would say "nobody left", which is not what "no prior year" means.

    python pipeline/officer_features.py             # report only
    python pipeline/officer_features.py --write     # patch train_matrix.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OFFICERS = ROOT / "pipeline" / "data" / "officers.json"
MATRIX = ROOT / "pipeline" / "data" / "train_matrix.npz"
MANIFEST = ROOT / "pipeline" / "data" / "feature_manifest.json"

FILLS = ("NEO_COUNT", "NEO_TURNOVER", "CEO_DUALITY", "CEO_TENURE")
CANNOT = (
    "CEO_AGE",
    "CEO_FOUNDER_FLAG",
    "CEO_EQUITY_PCT",
    "CEO_PAY_RATIO",
    "BOARD_INDEP_PCT",
    "BOARD_SIZE",
    "INSIDER_OWN_PCT",
    "CEO_PAY_VS_SECTOR",
)


# Divisional/subsidiary CEO titles. Form 345 reports officers of the FILER, and large
# groups file for unit heads too: "Chairman & CEO Arch Insur Gr", "Reinsur. Group Chairman
# & CEO". Those people are not the company's chief executive, and counting them as such
# corrupts both tenure and duality.
DIVISIONAL = (
    "insur gr",
    "reinsur",
    " group ",
    " gr ",
    "division",
    "divisional",
    "subsidiary",
    " unit",
    "segment",
    "americas",
    "international",
    " emea",
    " apac",
    " na ",
)


def _is_divisional(title: str) -> bool:
    t = f" {(title or '').lower()} "
    return any(m in t for m in DIVISIONAL)


def ceo_of(rows) -> tuple[str | None, str, bool]:
    """(name, title, unambiguous) for the company's chief executive.

    ORDER-DEPENDENT SELECTION WAS A REAL DEFECT, not a tidiness issue. 1,197 of 4,975
    company-years (24.1%) carry MORE THAN ONE row whose role is CEO, and the first version
    of this file took whichever came first in the list. Two different things produce those
    duplicates and both were being silently resolved by list order:

        SUBSIDIARY HEADS  ACGL_2015 lists GRANDISSON ("Reinsur. Group Chairman & CEO")
                          alongside IORDANOU ("Chairman & CEO"). Only one runs Arch.
        TRANSITION YEARS  ABT_2020 lists Ford ("President and CEO") and WHITE
                          ("Chairman and CEO") — the handover year, both genuine.

    Divisional titles are filtered out because they are identifiably not the top job. What
    remains ambiguous — a real transition — is REPORTED AS AMBIGUOUS rather than guessed,
    and the caller masks tenure and duality for that row. Picking one arbitrarily would
    fabricate a CEO change whenever list order flipped, and CEO_TENURE is precisely the
    column that would absorb it as a genuine-looking number.
    """
    ceos = [o for o in rows if o.get("role") == "CEO" and o.get("name")]
    if not ceos:
        return None, "", False
    top = [o for o in ceos if not _is_divisional(o.get("title", ""))] or ceos
    names = {o["name"] for o in top}
    return top[0]["name"], top[0].get("title") or "", len(names) == 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    off = json.loads(OFFICERS.read_text(encoding="utf-8"))
    z = np.load(MATRIX, allow_pickle=True)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = [x if isinstance(x, str) else (x.get("feature") or x.get("name")) for x in man["features"]]
    Z, M = z["Z"].copy(), z["mask"].copy()
    tick = z["ticker"].tolist()
    year = [int(y) for y in z["fiscal_year"].tolist()]

    missing = [f for f in FILLS if f not in names]
    if missing:
        print(f"columns not in the manifest, refusing: {missing}")
        return 2
    col = {f: names.index(f) for f in FILLS}

    first_year = {}
    for t, y in zip(tick, year, strict=False):
        first_year[t] = min(y, first_year.get(t, y))

    # first year each (ticker, ceo) pair is seen, across the whole panel
    ceo_first: dict[tuple[str, str], int] = {}
    for k, rows in off.items():
        t, _, ys = k.rpartition("_")
        if not ys.isdigit():
            continue
        c, title, unambiguous = ceo_of(rows)
        if c and unambiguous:
            key = (t, c)
            ceo_first[key] = min(int(ys), ceo_first.get(key, int(ys)))

    filled = {f: 0 for f in FILLS}
    for i, (t, y) in enumerate(zip(tick, year, strict=False)):
        rows = off.get(f"{t}_{y}")
        if not rows:
            continue

        Z[i, col["NEO_COUNT"]] = float(len(rows))
        M[i, col["NEO_COUNT"]] = 1.0
        filled["NEO_COUNT"] += 1

        prev = off.get(f"{t}_{y-1}")
        if prev:
            a = {o["name"] for o in prev if o.get("name")}
            b = {o["name"] for o in rows if o.get("name")}
            if a:
                Z[i, col["NEO_TURNOVER"]] = float(len(a - b) / len(a))
                M[i, col["NEO_TURNOVER"]] = 1.0
                filled["NEO_TURNOVER"] += 1

        c, title, unambiguous = ceo_of(rows)
        if c and unambiguous:
            Z[i, col["CEO_DUALITY"]] = 1.0 if "chair" in title.lower() else 0.0
            M[i, col["CEO_DUALITY"]] = 1.0
            filled["CEO_DUALITY"] += 1

            fy = ceo_first.get((t, c))
            # Only when the transition was OBSERVED inside the panel. See docstring.
            if fy is not None and fy > first_year.get(t, fy):
                Z[i, col["CEO_TENURE"]] = float(y - fy)
                M[i, col["CEO_TENURE"]] = 1.0
                filled["CEO_TENURE"] += 1

    n = len(tick)
    print(f"company-years in matrix: {n}\n")
    print(f"  {'column':16} {'filled':>7} {'coverage':>9}   note")
    for f in FILLS:
        note = ""
        if f == "CEO_TENURE":
            note = "left-censored; only observed transitions"
        elif f == "NEO_TURNOVER":
            note = "needs a prior year"
        print(f"  {f:16} {filled[f]:>7} {100*filled[f]/n:>8.1f}%   {note}")
    print("\n  still zero-coverage and NOT fillable from Form 345 reporting owners:")
    print(f"    {', '.join(CANNOT)}")
    print(
        "    (BOARD_INDEP_PCT / BOARD_SIZE need DIRECTORS, excluded from this source "
        "by design — not a gap this route can ever close)"
    )

    dead_before = int((z["mask"].mean(axis=0) == 0).sum())
    dead_after = int((M.mean(axis=0) == 0).sum())
    print(f"\n  zero-coverage columns: {dead_before} -> {dead_after}")
    print(f"  matrix mean observed : {100*z['mask'].mean():.1f}% -> {100*M.mean():.1f}%")

    if args.write:
        out = {k: z[k] for k in z.files}
        out["Z"], out["mask"] = Z, M
        np.savez(MATRIX, **out)
        print(f"\nwrote {MATRIX}")
    else:
        print("\ndry run — pass --write to patch the matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
