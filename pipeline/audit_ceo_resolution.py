"""officer_features.ceo_of() names a DIVISIONAL executive as CEO in 268 ticker-years.

CEO_DUALITY and CEO_TENURE are trained features in the equities MTNN and both depend on
correctly identifying WHICH person is the chief executive of the registrant. ceo_of()
carries a DIVISIONAL stoplist -- ("insur gr", "reinsur", " group ", " gr ", "division")
-- written against the insurers it was first run on. It does not generalise.

The failure surfaced in vector-unified: for FDX it returned Krishnasamy Sriram, whose
title is "EVP CDI Off & CTO/CEO FDW" -- FedEx Dataworks -- while the actual CEO,
Subramaniam Rajesh, sat in the same rows as "President/CEO". That prompted this audit,
which compares ceo_of() against a rule that needs no stoplist at all:

    A CORPORATE CEO'S TITLE IS MADE ONLY OF ROLE WORDS.
    A DIVISIONAL CEO'S TITLE NAMES THE DIVISION.

No list of division names, so nothing to keep up to date and nothing tuned against the
cases it judges.

RESULT over 5,352 ticker-years:

    both name a CEO and agree                 4,309
    both name a CEO and DISAGREE                268   ceo_of picks the divisional one
    only ceo_of names one                       398   see the split below
    only the role-word rule names one             0   strictly more conservative
    neither                                     377

And the blast radius on the TRAINED feature is larger than the identity
disagreement, because CEO_TENURE is years-since-first-seen and one wrong identity
shifts every later year for that ticker:

    CEO_TENURE rows written by both           2,744
    ...and DIFFERENT under the two rules        558   = 20.3%

Every disagreement sampled runs the same way. Accenture: ceo_of says Casati Gianfranco,
"CEO-Growth Markets"; the corporate CEO is Sweet Julie Spellman, "Chair & CEO". Amazon
2022: ceo_of says Clark David H, "CEO Worldwide Consumer"; it is Jassy Andrew R,
"President and CEO". AIG: ceo_of says Hogan Kevin T., "CEO, Corebridge Financial"; it is
Zaffino Peter, "Chairman & CEO". American Tower: Font Juan, "SVP, Pres. & CEO, CoreSite"
against Vondran Steven O, "President and CEO".

THE REFUSALS ARE NOT ALL ceo_of BEING RIGHT AND THE ROLE-WORD RULE BEING TOO STRICT.
An earlier version of this rule refused EVERY multi-candidate year, 717 of them. They
split three ways, and the middle group is the interesting one:

    319  several pure-role rows, ALL THE SAME PERSON. "Chair and CEO" plus "Chairman and
         CEO" for one human. Resolvable by deduplicating on name, and the rule below now
         does that.
    297  several pure-role rows, DIFFERENT people. These are CEO TRANSITION YEARS.
         ACN_2019 carries Nanterme Pierre, Rowland David and Sweet Julie Spellman -- the
         CEO died in January, an interim served, a successor took over in September. All
         three are real. ABT_2020 has White Miles D handing to Ford Robert B. AEP_2024
         has an interim plus two others.
    101  no pure-role row at all; every CEO-role candidate is divisional.

The 297 matter more than the 268. A year with two chief executives has no single answer,
and CEO_TENURE is precisely the feature that is supposed to notice. Picking one
arbitrarily does not merely risk being wrong -- it erases the transition, which is the
signal. ceo_of() returns a name for all 297 and flags them, but a downstream consumer
that reads the name and not the flag sees a stable CEO where there was a succession.

THIS SCRIPT CHANGES NOTHING. Fixing ceo_of() moves a trained model input and the shipped
equities model was fit with the current behaviour. The decision of whether to re-fit
belongs to the operator; this produces the number it should be made on.

    python pipeline/audit_ceo_resolution.py

Writes: pipeline/data/ceo_resolution_audit.json
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICERS = ROOT / "pipeline" / "data" / "officers.json"
OFFICER_FEATURES = ROOT / "pipeline" / "officer_features.py"
OUT = ROOT / "pipeline" / "data" / "ceo_resolution_audit.json"

ROLE_WORDS = {
    "chairman",
    "chairwoman",
    "chair",
    "president",
    "ceo",
    "chief",
    "executive",
    "officer",
    "co",
    "and",
    "the",
    "of",
    "&",
    "-",
    "director",
    "founder",
    "interim",
}


def pure_role_rows(rows):
    """CEO-role rows whose title contains only role words -- no division named."""
    out = []
    for r in rows:
        if (r.get("role") or "") != "CEO":
            continue
        toks = [t for t in re.split(r"[^A-Za-z&\-]+", str(r.get("title", "")).lower()) if t]
        if toks and all(t in ROLE_WORDS for t in toks):
            out.append(r)
    return out


def corporate_ceo(rows):
    """(name, title, how). Same rule as vector-unified/pipeline/build_bridge_index.py,
    plus the name-dedup refinement this audit showed was needed: several pure-role rows
    naming ONE person is a title-spelling artifact, not ambiguity."""
    ceos = [r for r in rows if (r.get("role") or "") == "CEO"]
    if not ceos:
        return None, None, "absent"
    pure = pure_role_rows(rows)
    if len(pure) == 1:
        return pure[0]["name"], pure[0]["title"], "exact"
    if len(pure) > 1:
        names = {r["name"].strip().upper() for r in pure}
        if len(names) == 1:
            return pure[0]["name"], pure[0]["title"], "same_person_two_titles"
        return None, None, "transition_or_multiple"
    if len(ceos) == 1:
        return ceos[0]["name"], ceos[0]["title"], "sole_ceo_row"
    return None, None, "all_candidates_divisional"


def tenure_series(by_ticker, pick):
    """{(ticker, year): tenure} under officer_features' own left-censoring rule --
    written only when the CEO's first appearance is AFTER the company's first panel
    year, i.e. when a transition was actually observed."""
    out = {}
    for t, yr in by_ticker.items():
        years = sorted(yr)
        if not years:
            continue
        first_panel = years[0]
        first_seen = {}
        for y in years:
            n, _, _ = pick(yr[y])
            if not n:
                continue
            n = n.strip().upper()
            first_seen.setdefault(n, y)
            if first_seen[n] > first_panel:
                out[(t, y)] = y - first_seen[n]
    return out


def main() -> int:
    if not OFFICERS.exists() or not OFFICER_FEATURES.exists():
        print("FAIL: officers.json or officer_features.py missing", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("officer_features", OFFICER_FEATURES)
    of = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(of)

    d = json.loads(OFFICERS.read_text(encoding="utf-8"))
    agree = disagree = only_eq = only_me = neither = 0
    how_counts = Counter()
    disagreements = []
    transitions = []

    for k, rows in d.items():
        a_name, a_title, _ = of.ceo_of(rows)
        b_name, b_title, how = corporate_ceo(rows)
        how_counts[how] += 1
        if a_name and b_name:
            if a_name.strip().upper() == b_name.strip().upper():
                agree += 1
            else:
                disagree += 1
                if len(disagreements) < 40:
                    disagreements.append(
                        {
                            "key": k,
                            "ceo_of_says": a_name,
                            "ceo_of_title": a_title,
                            "role_word_rule_says": b_name,
                            "role_word_title": b_title,
                        }
                    )
        elif a_name:
            only_eq += 1
        elif b_name:
            only_me += 1
        else:
            neither += 1
        if how == "transition_or_multiple" and len(transitions) < 25:
            transitions.append(
                {
                    "key": k,
                    "people": [f"{r['name']} | {r['title']}" for r in pure_role_rows(rows)],
                    "ceo_of_picked": a_name,
                }
            )

    out = {
        "question": "How often does officer_features.ceo_of() name a divisional " "executive as the registrant's CEO?",
        "why_it_matters": "CEO_DUALITY and CEO_TENURE are trained features in the "
        "equities MTNN and both depend on identifying the right "
        "person. A divisional CEO changes far more often than a "
        "corporate one, so a wrong pick corrupts tenure most.",
        "ticker_years": len(d),
        "both_name_a_ceo_and_agree": agree,
        "both_name_a_ceo_and_DISAGREE": disagree,
        "disagreement_rate_pct": round(100.0 * disagree / max(agree + disagree, 1), 2),
        "only_ceo_of_names_one": only_eq,
        "only_role_word_rule_names_one": only_me,
        "neither": neither,
        "role_word_rule_resolution_counts": dict(how_counts.most_common()),
        "direction_of_every_sampled_disagreement": "ceo_of names the DIVISIONAL "
        "executive; the role-word rule names the corporate one. Accenture -> "
        "'CEO-Growth Markets' over Sweet Julie Spellman 'Chair & CEO'. Amazon 2022 "
        "-> 'CEO Worldwide Consumer' over Jassy Andrew R 'President and CEO'. AIG -> "
        "'CEO, Corebridge Financial' over Zaffino Peter 'Chairman & CEO'.",
        "why_ceo_of_misses_them": "Its DIVISIONAL stoplist is ('insur gr', 'reinsur', "
        "' group ', ' gr ', 'division') -- written against the insurers it was first "
        "run on. It cannot see 'Growth Markets', 'Worldwide Consumer', 'Corebridge', "
        "'CoreSite' or 'FDW'. A stoplist of division NAMES can never be complete; "
        "the role-word rule needs no such list because it tests what a corporate "
        "title IS rather than what a divisional one contains.",
        "the_297_transitions": {
            "count": how_counts.get("transition_or_multiple", 0),
            "what_they_are": "Ticker-years with more than one person holding a pure "
            "corporate CEO title -- successions. ACN_2019 carries Nanterme Pierre, "
            "Rowland David and Sweet Julie Spellman: the CEO died in January, an "
            "interim served, a successor took over in September. All three are real.",
            "why_they_matter_more_than_the_247": "A year with two chief executives has "
            "no single answer, and CEO_TENURE is exactly the feature meant to notice. "
            "Picking one arbitrarily does not merely risk being wrong, it ERASES the "
            "transition -- which is the signal. ceo_of() returns a name for all of "
            "them; a consumer reading the name and not the flag sees a stable CEO "
            "where there was a succession.",
            "examples": transitions,
        },
        "changes_nothing": "Fixing ceo_of() moves a trained model input and the shipped "
        "equities model was fit with the current behaviour. Whether to re-fit is an "
        "operator decision; this produces the number to make it on.",
        "disagreement_examples": disagreements,
    }

    # --- blast radius on the actual trained feature -------------------------
    # "a divisional CEO changes more often, so tenure is corrupted most" was an
    # assertion until this ran. It is 20.3%, not 5.86% -- a wrong identity does not
    # cost one row, it shifts the whole first-seen series for that ticker.
    by_ticker = {}
    for k, rows in d.items():
        t, _, y = k.rpartition("_")
        if y.isdigit():
            by_ticker.setdefault(t, {})[int(y)] = rows
    ta = tenure_series(by_ticker, of.ceo_of)
    tb = tenure_series(by_ticker, corporate_ceo)
    shared = set(ta) & set(tb)
    differ = [k for k in shared if ta[k] != tb[k]]
    out["CEO_TENURE_blast_radius"] = {
        "why": "The identity disagreement is 5.86% of ticker-years. That is NOT the "
        "blast radius on the trained feature, because CEO_TENURE is years since "
        "the CEO was first seen -- one wrong identity shifts every later year for "
        "that ticker. Measured rather than asserted.",
        "rows_written_by_ceo_of": len(ta),
        "rows_written_by_role_word_rule": len(tb),
        "written_by_both": len(shared),
        "written_by_both_and_DIFFERENT": len(differ),
        "pct_of_shared_rows_that_change": round(100.0 * len(differ) / max(len(shared), 1), 1),
        "only_ceo_of_writes": len(set(ta) - set(tb)),
        "only_role_word_rule_writes": len(set(tb) - set(ta)),
        "examples": [
            {"key": f"{t}_{y}", "ceo_of": ta[(t, y)], "role_word_rule": tb[(t, y)]} for (t, y) in sorted(differ)[:20]
        ],
        "reading": "20.3% of the CEO_TENURE values that get written change under the "
        "corrected identity. Most are off-by-one shifts -- the two rules "
        "disagree about WHEN the current CEO started -- but ACN_2023 goes 0 "
        "to 3, a three-year error in a feature whose whole content is elapsed "
        "time. This is the number the re-fit decision should be made on, not "
        "the 5.86%.",
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"ticker-years                        {len(d)}")
    print(f"  both name a CEO and agree         {agree}")
    print(f"  both name a CEO and DISAGREE      {disagree}  " f"({out['disagreement_rate_pct']}%)")
    print(f"  only ceo_of names one             {only_eq}")
    print(f"  only role-word rule names one     {only_me}")
    print(f"  neither                           {neither}")
    print(f"\nrole-word rule resolution: {dict(how_counts.most_common())}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
