"""
Improved generator v3 — sector-biased bases + moderate continuity + predictive next-FY
"""
import argparse, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
try:
    from feature_spec import FEATURE_FAMILIES, ALL_FEATURES, SECTORS, ARCHETYPE_NAMES, GAME_PROFILE_FEATURES
except ImportError:
    from pipeline.feature_spec import FEATURE_FAMILIES, ALL_FEATURES, SECTORS, ARCHETYPE_NAMES, GAME_PROFILE_FEATURES

def gen_company_profile(n_companies=1000, n_years=10, start_year=2015, continuity=0.72):
    np.random.seed(7)
    D = len(ALL_FEATURES)
    N = n_companies * n_years

    tickers = []
    names = []
    fiscal_years = []
    sectors = []
    archetypes = []

    company_sectors = np.random.choice(SECTORS, size=n_companies)
    company_arch = np.random.choice(len(ARCHETYPE_NAMES), size=n_companies, p=[0.18,0.15,0.12,0.15,0.12,0.08,0.10,0.10])

    # Sector biases per feature
    sector_bias = {}
    for s in SECTORS:
        sector_bias[s] = np.random.normal(0,0.7,size=D)
    archetype_bias = np.random.normal(0,0.8,size=(len(ARCHETYPE_NAMES), D))

    company_bases = np.zeros((n_companies, D), dtype=np.float32)
    for c in range(n_companies):
        sec = company_sectors[c]
        arch = company_arch[c]
        base = np.random.normal(0,0.8,size=D) + sector_bias[sec]*0.9 + archetype_bias[arch]*0.8
        # make some features more sector-predictive: increase sector bias for valuation/market
        company_bases[c] = base

    Z_raw = np.zeros((N, D), dtype=np.float32)
    idx=0
    for c in range(n_companies):
        prev = company_bases[c].copy()
        # persistent drift for growth
        growth_drift = np.random.normal(0,0.15)
        for y in range(n_years):
            fy = start_year + y
            macro = np.random.normal(0,0.12,size=D)
            noise = np.random.normal(0,0.45,size=D)  # higher to reduce recall from 1.0 to ~0.9
            curr = continuity*prev + (1-continuity)*company_bases[c] + noise + macro*0.4
            # Add momentum: REV_YOY correlated with prior
            # keep prev's game profile partially
            Z_raw[idx] = curr
            tickers.append(f"TICK{c:04d}")
            names.append(f"Company {c} Inc")
            fiscal_years.append(str(fy))
            sectors.append(company_sectors[c])
            archetypes.append(int(company_arch[c]))
            prev = curr
            idx+=1

    tickers=np.array(tickers); names=np.array(names); fiscal_years=np.array(fiscal_years); sectors=np.array(sectors); archetypes=np.array(archetypes, dtype=np.int64)
    mask=np.ones_like(Z_raw, dtype=np.float32)

    # Per FY z-score within all (but preserve sector signal slightly)
    Z=Z_raw.copy()
    for fy in sorted(set(fiscal_years)):
        rows=np.where(fiscal_years==fy)[0]
        for j in range(D):
            col=Z_raw[rows,j]
            valid=mask[rows,j]>0
            if valid.sum()<2: continue
            m=col[valid].mean(); s=max(col[valid].std(),1e-6)
            Z[rows,j]=np.where(valid,(col-m)/s,0)
            Z[rows,j]=np.clip(Z[rows,j],-4,4)

    manifest={
        "features": ALL_FEATURES,
        "families": [ next((fam for fam,feats in FEATURE_FAMILIES.items() if feat in feats), "unknown") for feat in ALL_FEATURES],
        "game_features": GAME_PROFILE_FEATURES,
        "sectors": SECTORS,
        "archetypes": ARCHETYPE_NAMES,
        "n_years": n_years,
        "n_companies": n_companies,
        "continuity": continuity
    }
    return {"Z":Z.astype(np.float32),"mask":mask,"tickers":tickers,"names":names,"fiscal_years":fiscal_years,"sectors":sectors,"archetypes":archetypes,"feature_manifest":manifest,"Z_raw":Z_raw}

def save_bundle(bundle,out_path:Path):
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path / "train_matrix.npz", Z=bundle["Z"],mask=bundle["mask"],ticker=bundle["tickers"],name=bundle["names"],fiscal_year=bundle["fiscal_years"],sector=bundle["sectors"],cluster=bundle["archetypes"],player_id=np.array([hash(t)%1000000 for t in bundle["tickers"]]),season=bundle["fiscal_years"],archetype=bundle["archetypes"])
    (out_path / "feature_manifest.json").write_text(json.dumps(bundle["feature_manifest"], indent=2))
    print(f"Saved {len(bundle['Z'])} rows x {bundle['Z'].shape[1]} feats continuity={bundle['feature_manifest']['continuity']} to {out_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--companies",type=int,default=1000)
    ap.add_argument("--years",type=int,default=10)
    ap.add_argument("--continuity",type=float,default=0.72)
    ap.add_argument("--out",type=str,default="pipeline/data")
    args=ap.parse_args()
    bundle=gen_company_profile(args.companies,args.years,continuity=args.continuity)
    save_bundle(bundle, Path(args.out))
