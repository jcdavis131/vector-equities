"""Derive the map's coordinate file from the real embedding export.

assets/real_data.json is the model's own output: 4,831 fiscal-year rows with real
tickers, sectors, archetypes and PCA-3 coordinates. It is 4.4 MB because it carries
a 64-float embedding and 12 skill floats per row, which the map does not need.

This drops those two arrays and nothing else. Every value written here is copied
verbatim from the export; no value is computed, rounded into existence, or invented.
Re-run after any re-export:  python scripts/build_map_slim.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "real_data.json"
OUT = ROOT / "public" / "assets" / "map_slim.json"

raw = json.loads(SRC.read_text(encoding="utf-8"))
rows = raw if isinstance(raw, list) else next(v for v in raw.values() if isinstance(v, list))

keep = ("ticker", "name", "year", "sector", "archetype", "x", "y", "z")
out = [{k: r[k] for k in keep if k in r} for r in rows]
missing = [i for i, r in enumerate(out) if not all(k in r for k in ("x", "y", "z", "sector"))]
if missing:
    raise SystemExit(f"{len(missing)} rows lack coordinates or sector; refusing to write a partial map")

sectors = sorted({r["sector"] for r in out})
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "source": "assets/real_data.json",
    "n_rows": len(out),
    "sectors": sectors,
    "note": "coordinates and labels copied verbatim from the model export; emb/skills dropped for page weight",
    "rows": out,
}, separators=(",", ":")), encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} real rows, {len(sectors)} sectors, "
      f"{OUT.stat().st_size/1024:.0f} KB")
