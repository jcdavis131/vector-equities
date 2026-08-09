"""Export the shippable real_data.json from pipeline/data/embedding.npz.

train_mtnn.py (the README Quickstart's train step) already writes
embedding.npz with everything needed: 64-d embeddings + ticker/name/
fiscal_year/sector/cluster/skill_pred, one row per company-FY. This script
just does the PCA(3) projection and assembles the final points list --
critically, WITH the "emb" field included (assets/real_data.json currently
in this checkout doesn't have it; export_v6_real_assets.py's own generated
points do include it, so its absence in what's shipped means whatever run
produced the live file predates that field being added, or stripped it in a
step not present in this checkout -- see tasks/rebuild_2026-07-30_notes.md).

Run: python pipeline/export_real_assets.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
ASSETS_DIR = ROOT / "assets"


def main() -> int:
    d = np.load(DATA_DIR / "embedding.npz", allow_pickle=True)
    E = d["E"].astype(np.float32)
    tickers = d["ticker"].astype(str)
    names = d["name"].astype(str)
    fiscal_years = d["fiscal_year"]
    sectors = d["sector"].astype(str)
    clusters = d["cluster"]
    skill_pred = d["skill_pred"].astype(np.float32)
    skill_keys = [str(k) for k in d["skill_keys"]]

    with (DATA_DIR / "feature_manifest.json").open() as f:
        manifest = json.load(f)
    with (DATA_DIR / "mtnn_report.json").open() as f:
        report = json.load(f)

    norms = np.linalg.norm(E, axis=1)
    assert not np.isnan(E).any(), "NaN in embeddings"
    print(f"loaded {E.shape[0]} rows, {E.shape[1]}-d, norm mean={norms.mean():.4f}")

    pca = PCA(n_components=3, random_state=7)
    xyz = pca.fit_transform(E)
    W = pca.components_.astype(np.float32)

    # Canonical order/naming from build_archetypes.py -- must match exactly,
    # cluster ids are assigned against this list at build time.
    archetype_names = [
        "Compounder",
        "Cash_Cow",
        "Turnaround",
        "HyperGrowth_SaaS",
        "Heavy_Industrial",
        "Bank_Capital_Heavy",
        "Moonshot_Bio",
        "Serial_Acquirer",
    ]

    points = []
    for i in range(len(tickers)):
        arch_id = int(clusters[i]) if not np.isnan(clusters[i]) else 0
        points.append(
            {
                "ticker": str(tickers[i]),
                "name": str(names[i]),
                "year": str(fiscal_years[i]),
                "sector": str(sectors[i]),
                "archetype": archetype_names[arch_id % len(archetype_names)],
                "x": round(float(xyz[i, 0]), 5),
                "y": round(float(xyz[i, 1]), 5),
                "z": round(float(xyz[i, 2]), 5),
                "skills": [round(float(s), 4) for s in skill_pred[i]],
                "emb": [round(float(v), 5) for v in E[i]],
            }
        )
    points.sort(key=lambda p: (p["ticker"], p["year"]))

    provenance = {
        "sop": "data_provenance_SOP.md",
        "note": (
            "Field-level provenance per operator directive (real + verified data only). "
            "Rebuilt 2026-07-30 end to end: fetch_sec_summary.py (SEC EDGAR XBRL, real "
            "User-Agent fix) + fetch_market_history.py (yfinance, upgraded 0.2.18->1.5.2 "
            "after both were silently 100% failing) -> build_real_from_summary.py -> "
            "build_skills.py + build_archetypes.py -> train_mtnn.py --dim 64 --fusion "
            "transformer --d-model 128 -> this export. No synthetic rows."
        ),
        "fields": {
            "embeddings": {
                "classification": "REAL",
                "detail": "Model forward-pass 64-d embeddings from train_mtnn.py, all rows real SEC/market data, no "
                "placeholder rows.",
            },
            "skills": {
                "classification": "REAL",
                "detail": "Model skill_towers head predictions, trained on real feature matrix.",
            },
        },
    }

    real_data_obj = {
        "points": points,
        "skill_keys": skill_keys,
        "archetypes": archetype_names,
        "sectors": sorted(set(sectors.tolist())),
        "model": f"equities_mtnn_v_rebuild_d{E.shape[1]}_transformer",
        "dim": int(E.shape[1]),
        "fusion": "transformer",
        "rows": len(points),
        "tickers": len(set(tickers.tolist())),
        "years": sorted({str(y) for y in fiscal_years}),
        "features": len(manifest["features"]),
        "cqs": report.get("composite", {}).get("cqs"),
        "recall_at_10": report.get("composite", {}).get("recall_at_10"),
        "purity_at_20": report.get("composite", {}).get("purity_at_20"),
        "provenance": provenance,
        "proj": {
            "W": [[round(float(v), 6) for v in row] for row in W],
            "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        },
        "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS_DIR / "real_data.json"
    out_path.write_text(json.dumps(real_data_obj), encoding="utf-8")

    print(f"exported {out_path}  points={len(points)}  tickers={real_data_obj['tickers']}")
    print(
        f"PCA(3) explained variance: {real_data_obj['proj']['explained_variance']} "
        f"(sum {sum(real_data_obj['proj']['explained_variance']):.3f})"
    )
    print(
        f"CQS={real_data_obj['cqs']}  recall@10={real_data_obj['recall_at_10']}  "
        f"purity@20={real_data_obj['purity_at_20']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
