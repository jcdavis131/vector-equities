"""
Improved synthetic generator v2 — strong company identity for high recall
Company base vector persists + AR1 drift, ensuring same-ticker adjacent FY pairs are close in feature space before z-scoring.

This mirrors hoops: same-player adjacent seasons should be similar but evolving.
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

def gen_company_profile(n_companies=800, n_years=10, start_year=2015, continuity=0.85):
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
    # Company base embedding — strong signal
    company_bases = np.random.normal(0,1,size=(n_companies, D)).astype(np.float32)
    # Archetype bias on base
    arch_bias = np.random.normal(0,0.6,size=(len(ARCHETYPE_NAMES), D))
    for c in range(n_companies):
        company_bases[c] += arch_bias[company_arch[c]]*0.8

    # Time series per company: AR1 drift around base
    Z_raw = np.zeros((N, D), dtype=np.float32)
    idx=0
    for c in range(n_companies):
        prev = company_bases[c].copy()
        for y in range(n_years):
            fy = start_year + y
            # AR1: 0.85*prev + 0.15*base + small noise + macro
            macro = np.random.normal(0,0.08,size=D)
            noise = np.random.normal(0,0.25,size=D)  # yearly idiosyncratic, small vs base
            curr = continuity*prev + (1-continuity)*company_bases[c] + noise + macro*0.3
            # Enforce realistic constraints on key features via transformation
            # Map some dims to financial ranges but keep z-score identity
            # Example: make REV_YOY somewhat persistent: blend previous REV_YOY
            Z_raw[idx] = curr
            tickers.append(f"TICK{c:04d}")
            names.append(f"Company {c} Inc")
            fiscal_years.append(str(fy))
            sectors.append(company_sectors[c])
            archetypes.append(int(company_arch[c]))
            prev = curr
            idx+=1

    tickers = np.array(tickers)
    names = np.array(names)
    fiscal_years = np.array(fiscal_years)
    sectors = np.array(sectors)
    archetypes = np.array(archetypes, dtype=np.int64)

    # Mask — minimal missing
    mask = np.ones_like(Z_raw, dtype=np.float32)

    # Era z-score within FY (preserve relative rank, but company identity remains via rank)
    Z = Z_raw.copy()
    feat_to_idx = {f:i for i,f in enumerate(ALL_FEATURES)}
    for fy in sorted(set(fiscal_years)):
        rows = np.where(fiscal_years==fy)[0]
        for j in range(D):
            col = Z_raw[rows, j]
            valid = mask[rows, j]>0
            if valid.sum()<2:
                continue
            m = col[valid].mean()
            s = col[valid].std()
            s = max(s, 1e-6)
            Z[rows, j] = np.where(valid, (col - m)/s, 0)
            Z[rows, j] = np.clip(Z[rows, j], -4, 4)

    # Override some specific features to be more interpretable post-zscore (keep z but ensure archetype separation)
    # We already have archetype bias in base, so it's fine.

    manifest = {
        "features": ALL_FEATURES,
        "families": [ next((fam for fam,feats in FEATURE_FAMILIES.items() if feat in feats), "unknown") for feat in ALL_FEATURES],
        "game_features": GAME_PROFILE_FEATURES,
        "sectors": SECTORS,
        "archetypes": ARCHETYPE_NAMES,
        "n_years": n_years,
        "n_companies": n_companies,
        "continuity": continuity
    }
    return {
        "Z": Z.astype(np.float32),
        "mask": mask,
        "tickers": tickers,
        "names": names,
        "fiscal_years": fiscal_years,
        "sectors": sectors,
        "archetypes": archetypes,
        "feature_manifest": manifest,
        "Z_raw": Z_raw
    }

def save_bundle(bundle, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path / "train_matrix.npz",
        Z=bundle["Z"],
        mask=bundle["mask"],
        ticker=bundle["tickers"],
        name=bundle["names"],
        fiscal_year=bundle["fiscal_years"],
        sector=bundle["sectors"],
        cluster=bundle["archetypes"],
        player_id=np.array([hash(t) % 1000000 for t in bundle["tickers"]]),
        season=bundle["fiscal_years"],
        archetype=bundle["archetypes"]
    )
    (out_path / "feature_manifest.json").write_text(json.dumps(bundle["feature_manifest"], indent=2))
    print(f"Saved {len(bundle['Z'])} rows x {bundle['Z'].shape[1]} feats continuity={bundle['feature_manifest']['continuity']} to {out_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--companies", type=int, default=800)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--continuity", type=float, default=0.85)
    ap.add_argument("--out", type=str, default="pipeline/data")
    args=ap.parse_args()
    bundle = gen_company_profile(n_companies=args.companies, n_years=args.years, continuity=args.continuity)
    save_bundle(bundle, Path(args.out))
