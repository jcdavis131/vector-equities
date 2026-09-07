"""Derive the roster tiles from the real export - one company per sector.

WHY SECTOR AND NOT "ARCHETYPE": the export's `archetype` field is a k-means cluster index
dressed in a hardcoded name (pipeline/build_archetypes.py zips cluster i to
ARCHETYPE_NAMES[i]; nothing inspects a cluster's membership). The names are contradicted by
their own contents - "Bank_Capital_Heavy" is mostly Industrials, "HyperGrowth_SaaS" is 41%
of the dataset and leads with Financials. See FINDING_archetype_names_are_not_real.md.
Sector is a real, externally verifiable attribute of every row, so the page uses it.

SELECTION RULE (stated on the page, not hidden here): for each sector, take the FY2024 rows
in that sector and keep the one whose 64-d embedding is closest to that sector's mean
embedding. That is the most typical member of the sector in the space the model learned.

Every displayed value - ticker, name, sector, skill names and skill scores - is copied from
assets/real_data.json. Nothing is computed into existence.
Re-run after any re-export:  python scripts/build_roster.py
"""
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "real_data.json"
OUT = ROOT / "public" / "assets" / "roster.json"
YEAR = "2024"

raw = json.loads(SRC.read_text(encoding="utf-8"))
rows, skill_keys, sectors = raw["points"], raw["skill_keys"], raw["sectors"]

by_sector = {s: [r for r in rows if r.get("sector") == s] for s in sectors}
empty = [s for s, v in by_sector.items() if not v]
if empty:
    raise SystemExit(f"sectors with no rows: {empty}; refusing to write a partial roster")

tiles = []
for s in sectors:
    members = by_sector[s]
    dim = len(members[0]["emb"])
    centroid = [sum(r["emb"][i] for r in members) / len(members) for i in range(dim)]
    pool = [r for r in members if r["year"] == YEAR] or members
    best = min(pool, key=lambda r: sum((r["emb"][i] - centroid[i]) ** 2 for i in range(dim)))
    if len(best["skills"]) != len(skill_keys):
        raise SystemExit(f"{best['ticker']}: {len(best['skills'])} skills vs {len(skill_keys)} names")
    ranked = sorted(zip(skill_keys, best["skills"]), key=lambda kv: kv[1], reverse=True)
    tiles.append({
        "ticker": best["ticker"], "name": best["name"], "sector": s, "year": best["year"],
        "x": best["x"], "y": best["y"],
        "n_in_sector": len(members),
        "n_tickers_in_sector": len({r["ticker"] for r in members}),
        "dist_to_centroid": round(math.sqrt(sum((best["emb"][i] - centroid[i]) ** 2 for i in range(dim))), 4),
        "top_skills": [{"key": k, "value": round(v, 3)} for k, v in ranked[:2]],
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "source": "assets/real_data.json",
    "model": raw["model"],
    "selection": (f"per sector, the FY{YEAR} company whose 64-d embedding is nearest that "
                  "sector's mean embedding"),
    "skill_note": ("skill scores are the model's skill_towers head outputs, not percentages; "
                   "they are unbounded and shown as the model emits them"),
    "skill_keys": skill_keys,
    "tiles": tiles,
}, indent=1), encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}: {len(tiles)} tiles")
for t in tiles:
    sk = ", ".join(f"{k['key']} {k['value']:.2f}" for k in t["top_skills"])
    print(f"  {t['sector']:<24s} {t['ticker']:<6s} {t['name'][:28]:<28s} n={t['n_in_sector']:<5d} {sk}")
