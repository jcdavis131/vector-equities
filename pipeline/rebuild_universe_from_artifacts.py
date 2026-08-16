#!/usr/bin/env python3
"""Reconstruct pipeline/data/universe.json from artifacts already on disk.

WHY THIS EXISTS
---------------
`build_real_v5_career.py` reads `pipeline/data/universe.json` and dies without
it. That file was never tracked by git and is not on disk, so the matrix could
not be rebuilt at all (discovered 2026-08-15 while removing fabricated values
from the builder).

The obvious fix -- re-run `ticker_universe.py` -- fetches the CURRENT SEC ticker
list over the network. That silently changes which companies are in the
universe: index membership turns over, so a 2026 fetch does not reproduce the
500 companies the shipped matrix was actually built from. The rebuild would then
differ in its ROW SET as well as its values, and nothing downstream could tell
the two effects apart.

So this reconstructs the exact universe instead, and every field comes from a
real artifact:

  ticker, company, sector   the shipped matrix itself (train_matrix_v5.npz),
                            which was built from the universe we are recovering
  cik                       pipeline/data/expanded/universe_sec.json, the SEC
                            ticker->CIK mapping, tracked in git

Nothing is defaulted, guessed, or filled in. If a ticker cannot be resolved to a
CIK from a real source, this script FAILS rather than inventing one -- a
fabricated CIK would silently attach one company's filings to another.

The single ticker that `universe_sec.json` does not carry (AEP) is resolved from
its own cached SEC summary, whose `_meta.entity` reads "American Electric Power
Company, Inc." -- i.e. confirmed against the filer's own record, not assumed.

Usage:
    python pipeline/rebuild_universe_from_artifacts.py [--out pipeline/data/universe.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
SUMMARY_GLOB = str(ROOT / "pipeline" / "cache" / "sec" / "sec_summary" / "summary_*.json")


def load_matrix_universe(npz_path: Path) -> dict[str, tuple[str, str]]:
    """{ticker: (company, sector)} exactly as the shipped matrix records it."""
    d = np.load(npz_path, allow_pickle=True)
    for key in ("ticker", "name", "sector"):
        if key not in d:
            raise SystemExit(f"{npz_path.name}: missing '{key}' -- cannot recover the universe")
    out: dict[str, tuple[str, str]] = {}
    for t, n, s in zip(d["ticker"], d["name"], d["sector"], strict=False):
        out.setdefault(str(t), (str(n), str(s)))
    return out


def load_sec_ticker_map() -> dict[str, str]:
    """{TICKER: cik} from the tracked SEC mapping, keyed a few ways tickers vary."""
    p = DATA / "expanded" / "universe_sec.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- this is the only on-disk ticker->CIK source")
    by_t: dict[str, str] = {}
    for e in json.loads(p.read_text(encoding="utf-8")):
        t = str(e.get("ticker", "")).upper()
        cik = str(e.get("cik") or e.get("cik_str") or "")
        if t and cik:
            by_t.setdefault(t, cik)
    return by_t


def cached_summary_ciks() -> set[str]:
    return {os.path.basename(f)[len("summary_") : -len(".json")] for f in glob.glob(SUMMARY_GLOB)}


def resolve_cik(ticker: str, by_t: dict[str, str], summaries: set[str]) -> str | None:
    up = ticker.upper()
    for key in (up, up.replace("-", "."), up.replace(".", "-")):
        if key in by_t:
            return by_t[key]
    # Not in the SEC mapping. Fall back ONLY to a cached SEC summary whose own
    # _meta.entity we can read -- i.e. the filer's own record, still a real
    # source. Never invent a CIK.
    for cik in summaries:
        p = ROOT / "pipeline" / "cache" / "sec" / "sec_summary" / f"summary_{cik}.json"
        try:
            meta = json.loads(p.read_text(encoding="utf-8")).get("_meta", {})
        except Exception:  # noqa: BLE001
            continue
        entity = str(meta.get("entity", ""))
        if entity and _entity_matches(ticker, entity):
            print(f"  {ticker}: resolved via cached SEC summary -> CIK {cik} ({entity})")
            return cik
    return None


# Tickers whose company name we can confirm against a filer record. Kept
# explicit and tiny on purpose: this is a confirmation step, not a guessing one.
_ENTITY_HINTS = {"AEP": "american electric power"}


def _entity_matches(ticker: str, entity: str) -> bool:
    hint = _ENTITY_HINTS.get(ticker.upper())
    return bool(hint) and hint in entity.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(DATA / "train_matrix_v5.npz"))
    ap.add_argument("--out", default=str(DATA / "universe.json"))
    a = ap.parse_args()

    matrix_path = Path(a.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"missing {matrix_path} -- nothing to recover the universe from")

    uni_src = load_matrix_universe(matrix_path)
    by_t = load_sec_ticker_map()
    summaries = cached_summary_ciks()
    print(f"matrix universe {len(uni_src)} tickers | SEC map {len(by_t)} | cached summaries {len(summaries)}")

    universe, unresolved, no_summary = [], [], []
    for ticker, (company, sector) in sorted(uni_src.items()):
        cik = resolve_cik(ticker, by_t, summaries)
        if cik is None:
            unresolved.append(ticker)
            continue
        if str(cik).zfill(10) not in summaries:
            no_summary.append(ticker)
        universe.append(
            {
                "ticker": ticker,
                "cik": str(cik),
                "company": company,
                "sector": sector,
                "_provenance": "ticker/company/sector from train_matrix_v5.npz; cik from expanded/universe_sec.json",
            }
        )

    if unresolved:
        print(f"\nFAILED: {len(unresolved)} ticker(s) have no CIK in any real source: {unresolved}")
        print("Refusing to write a universe with invented identifiers.")
        return 1
    if no_summary:
        print(f"note: {len(no_summary)} ticker(s) resolved but have no cached SEC summary; "
              f"the builder skips those rows on its own: {no_summary[:8]}")

    Path(a.out).write_text(json.dumps(universe, indent=2), encoding="utf-8")
    print(f"\nWrote {len(universe)} entries -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
