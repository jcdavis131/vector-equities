"""Real CEO_TOTAL_COMP / AVG_NEO_COMP from pipeline/data/def14a_comp.json
(parse_def14a_xbrl.py). Mirrors market_features.py's get_market_row shape:
one lookup function, honest None when no filing matched (ticker/year not in
the fetched+parsed set), never an invented value.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP_JSON = ROOT / "pipeline" / "data" / "def14a_comp.json"

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        if COMP_JSON.exists():
            _cache = json.loads(COMP_JSON.read_text(encoding="utf-8"))
        else:
            _cache = {}
    return _cache


def get_def14a_row(ticker: str, year: int) -> dict:
    row = _load().get(f"{ticker}_{year}")
    if not row:
        return {"CEO_TOTAL_COMP": None, "AVG_NEO_COMP": None}
    return {
        "CEO_TOTAL_COMP": row.get("CEO_TOTAL_COMP"),
        "AVG_NEO_COMP": row.get("AVG_NEO_COMP"),
    }
