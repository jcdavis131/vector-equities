"""
Synthetic data generator for Vector Equities — offline-safe, realistic financial time series.

Generates train_matrix.npz compatible with vector-hoops pattern:
Z [N, D] z-scored per FY within sector, mask [N, D], names, tickers, FY, archetype labels, sector labels, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

try:
    from feature_spec import (
        ALL_FEATURES,
        ARCHETYPE_NAMES,
        FEATURE_FAMILIES,
        GAME_PROFILE_FEATURES,
        SECTORS,
    )
except ImportError:
    from pipeline.feature_spec import (
        ALL_FEATURES,
        ARCHETYPE_NAMES,
        FEATURE_FAMILIES,
        GAME_PROFILE_FEATURES,
        SECTORS,
    )

np.random.seed(7)


def gen_company_profile(n_companies=800, n_years=10, start_year=2015):
    """Generate correlated financials with archetype priors."""
    N = n_companies * n_years
    tickers = [f"TICK{i:04d}" for i in range(n_companies) for _ in range(n_years)]
    names = [f"Company {i} Inc" for i in range(n_companies) for _ in range(n_years)]
    fiscal_years = []
    sectors = []
    archetypes = []
    # Assign static sector & archetype per company
    company_sectors = np.random.choice(SECTORS, size=n_companies)
    company_arch = np.random.choice(
        len(ARCHETYPE_NAMES),
        size=n_companies,
        p=[0.18, 0.15, 0.12, 0.15, 0.12, 0.08, 0.10, 0.10],
    )

    # Base company quality factors — persistent
    company_quality = np.random.normal(0, 1, size=n_companies)  # latent
    company_growth_bias = np.random.normal(0, 0.5, size=n_companies)
    company_leverage_bias = np.random.normal(0, 0.5, size=n_companies)

    for c in range(n_companies):
        for y in range(n_years):
            fy = start_year + y
            fiscal_years.append(f"{fy}")
            sectors.append(company_sectors[c])
            archetypes.append(int(company_arch[c]))

    fiscal_years = np.array(fiscal_years)
    sectors = np.array(sectors)
    archetypes = np.array(archetypes, dtype=np.int64)
    tickers = np.array(tickers)
    names = np.array(names)

    # Features matrix raw (before z-score)
    D = len(ALL_FEATURES)
    Z_raw = np.zeros((N, D), dtype=np.float32)
    feat_to_idx = {f: i for i, f in enumerate(ALL_FEATURES)}

    # Helper to get company index per row
    comp_idx_per_row = np.repeat(np.arange(n_companies), n_years)

    # Generate year effects (macro regime)
    year_effects = {
        fy: np.random.normal(0, 0.2) for fy in range(start_year, start_year + n_years)
    }

    for row in range(N):
        c = comp_idx_per_row[row]
        fy = int(fiscal_years[row])
        qual = company_quality[c]
        grow_bias = company_growth_bias[c]
        lev_bias = company_leverage_bias[c]
        arch = company_arch[c]
        y_effect = year_effects[fy]

        # Archetype adjustments
        arch_adj = {
            0: {"margin": 0.5, "growth": 0.2, "debt": -0.3},  # Compounder
            1: {"margin": 0.8, "growth": -0.2, "debt": -0.5},  # Cash Cow
            2: {"margin": -0.6, "growth": -0.3, "debt": 0.6},  # Turnaround
            3: {"margin": -0.2, "growth": 0.9, "debt": -0.4},  # HyperGrowth SaaS
            4: {"margin": 0.0, "growth": 0.1, "debt": 0.2},  # Heavy Industrial
            5: {"margin": 0.1, "growth": 0.0, "debt": 0.8},  # Bank
            6: {"margin": -1.0, "growth": 0.5, "debt": -0.2},  # Moonshot Bio
            7: {"margin": 0.2, "growth": 0.5, "debt": 0.3},  # Serial Acquirer
        }[arch]

        # Income
        rev_yoy = np.random.normal(
            0.08 + grow_bias * 0.05 + arch_adj["growth"] * 0.05 + y_effect * 0.02, 0.18
        )
        gross_margin = np.clip(
            np.random.normal(0.40 + qual * 0.08 + arch_adj["margin"] * 0.1, 0.18),
            0.05,
            0.95,
        )
        op_margin = np.clip(gross_margin - np.random.normal(0.20, 0.10), -0.5, 0.6)
        net_margin = np.clip(
            op_margin - np.random.normal(0.05, 0.08) - max(0, lev_bias * 0.02),
            -0.8,
            0.5,
        )
        ebitda_margin = np.clip(op_margin + np.random.normal(0.08, 0.05), -0.3, 0.7)

        # Balance
        asset_turn = np.clip(np.random.normal(0.8 + qual * 0.1, 0.4), 0.1, 3.0)
        roa = net_margin * asset_turn * 0.6
        roe = roa * np.random.normal(1.8 + lev_bias * 0.3, 0.6)
        roic = np.clip(np.random.normal(roe * 0.8, 0.05), -0.3, 0.5)

        debt_to_equity = np.clip(
            np.random.normal(0.6 + lev_bias * 0.5 + arch_adj["debt"] * 0.3, 0.6),
            0.0,
            3.5,
        )
        debt_to_ebitda = (
            np.clip(np.random.normal(2.0 + lev_bias * 0.6, 1.5), 0.0, 8.0)
            if ebitda_margin > 0
            else np.random.normal(5, 2)
        )
        current_ratio = np.clip(np.random.normal(1.8 - lev_bias * 0.2, 0.6), 0.5, 5.0)

        # Cashflow
        fcf_margin = np.clip(
            np.random.normal(net_margin + np.random.normal(0.03, 0.05), 0.10), -0.5, 0.6
        )
        ocf_to_net = np.random.normal(1.1, 0.3)

        # Market
        ret_12m = np.random.normal(
            0.10 + qual * 0.08 + rev_yoy * 0.3 + y_effect * 0.05, 0.40
        )
        ret_6m = ret_12m * 0.6 + np.random.normal(0, 0.15)
        ret_3m = ret_12m * 0.3 + np.random.normal(0, 0.10)
        ret_1m = ret_12m * 0.08 + np.random.normal(0, 0.08)
        vol = np.clip(np.random.normal(0.32 - qual * 0.03, 0.12), 0.10, 1.2)
        beta = np.clip(np.random.normal(1.0 + lev_bias * 0.1, 0.35), 0.2, 2.5)

        # Valuation
        # Growth companies get higher multiples
        pe_base = 18 + grow_bias * 5 - net_margin * -10 + np.random.normal(0, 8)
        pe = np.clip(pe_base, 3, 80) if net_margin > 0 else np.random.normal(40, 20)
        pb = np.clip(np.random.normal(2.5 + roe * 5, 1.8), 0.3, 15)
        ev_ebitda = (
            np.clip(np.random.normal(12 + grow_bias * 2 + (1 - roic) * -5, 6), 2, 40)
            if ebitda_margin > 0
            else np.random.normal(20, 10)
        )

        # Management
        ceo_tenure = np.clip(np.random.normal(6 + qual * 1.2, 4), 0.5, 30)
        ceo_age = np.clip(ceo_tenure + np.random.normal(45, 6), 32, 78)
        founder = 1 if np.random.rand() < (0.25 if arch in [0, 3, 6] else 0.08) else 0
        ceo_comp = np.random.lognormal(mean=np.log(5e6) + qual * 0.2, sigma=0.8)  # $
        insider_own = np.clip(np.random.beta(2, 20) * 100 + founder * 15, 0.1, 60)

        # Ownership
        inst_pct = np.clip(np.random.normal(72, 15), 10, 95)
        insider_net = np.random.normal(0, 2) + (founder * 0.5)

        # Disclosure
        mda_sent = np.random.normal(0.05 + qual * 0.05, 0.3)
        risk_count = int(np.clip(np.random.normal(30 + lev_bias * 5, 8), 10, 90))

        # Assign to matrix
        # Use simplified mapping — we fill many features with transformations
        vals = {
            "REV_YOY": rev_yoy,
            "GROSS_MARGIN": gross_margin,
            "OP_MARGIN": op_margin,
            "NET_MARGIN": net_margin,
            "EBITDA_MARGIN": ebitda_margin,
            "ROE": roe,
            "ROA": roa,
            "ROIC": roic,
            "DEBT_TO_EQUITY": debt_to_equity,
            "DEBT_TO_EBITDA": debt_to_ebitda,
            "CURRENT_RATIO": current_ratio,
            "ASSET_TURNOVER": asset_turn,
            "FCF_MARGIN": fcf_margin,
            "OCF_TO_NET": ocf_to_net,
            "RET_12M": ret_12m,
            "RET_6M": ret_6m,
            "RET_3M": ret_3m,
            "RET_1M": ret_1m,
            "VOL_252D": vol,
            "VOL_90D": vol * np.random.normal(1, 0.15),
            "VOL_30D": vol * np.random.normal(1, 0.2),
            "BETA_1Y": beta,
            "PE": pe,
            "PB": pb,
            "EV_EBITDA": ev_ebitda,
            "CEO_TENURE": ceo_tenure,
            "CEO_AGE": ceo_age,
            "CEO_FOUNDER_FLAG": float(founder),
            "CEO_TOTAL_COMP": np.log(ceo_comp),
            "INSIDER_OWN_PCT": insider_own,
            "INST_PCT": inst_pct,
            "INSIDER_NET_12M": insider_net,
            "MDA_SENTIMENT": mda_sent,
            "RISK_FACTOR_COUNT": float(risk_count),
            "REV_3Y_CAGR": rev_yoy * 0.8 + np.random.normal(0, 0.06),
            "EBITDA": ebitda_margin * 100,
            "NET_INCOME": net_margin * 100,
        }
        # Fill rest with noise around quality
        for feat in ALL_FEATURES:
            idx = feat_to_idx[feat]
            if feat in vals:
                Z_raw[row, idx] = vals[feat]
            else:
                # generic correlated noise
                Z_raw[row, idx] = np.random.normal(qual * 0.2, 1.0)

        # Sector-adj extras
        # Force sector one-hot proxy via features already

    # Mask — simulate missing for some families early years
    mask = np.ones_like(Z_raw, dtype=np.float32)
    # Pre-2018 disclosure sparse
    for i, fy in enumerate(fiscal_years):
        if int(fy) < 2018:
            for feat in ["MDA_SENTIMENT", "RISK_FACTOR_COUNT", "TONE"]:
                if feat in feat_to_idx:
                    if np.random.rand() < 0.3:
                        mask[i, feat_to_idx[feat]] = 0
    # Banks lack some
    for i, sec in enumerate(sectors):
        if sec == "Financials" and np.random.rand() < 0.2:
            mask[i, feat_to_idx["INVENTORY_TURNOVER"]] = 0

    # Era z-score within fiscal year (sector-neutral like hoops)
    Z = Z_raw.copy()
    for fy in sorted(set(fiscal_years)):
        rows = np.where(fiscal_years == fy)[0]
        for j in range(D):
            col = Z_raw[rows, j]
            valid = mask[rows, j] > 0
            if valid.sum() < 2:
                continue
            m = col[valid].mean()
            s = col[valid].std()
            s = max(s, 1e-6)
            Z[rows, j] = np.where(valid, (col - m) / s, 0)
            # clip ±4 like hoops
            Z[rows, j] = np.clip(Z[rows, j], -4, 4)

    # Build manifest
    families = []
    for feat in ALL_FEATURES:
        # find family
        for fam, feats in FEATURE_FAMILIES.items():
            if feat in feats:
                families.append(fam)
                break
        else:
            families.append("unknown")

    manifest = {
        "features": ALL_FEATURES,
        "families": families,
        "game_features": GAME_PROFILE_FEATURES,
        "sectors": SECTORS,
        "archetypes": ARCHETYPE_NAMES,
        "n_years": n_years,
        "n_companies": len(set(tickers)),
    }

    return {
        "Z": Z.astype(np.float32),
        "mask": mask.astype(np.float32),
        "tickers": tickers,
        "names": names,
        "fiscal_years": fiscal_years,
        "sectors": sectors,
        "archetypes": archetypes,
        "feature_manifest": manifest,
        "Z_raw": Z_raw,
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
        player_id=np.array(
            [hash(t) % 100000 for t in bundle["tickers"]]
        ),  # hoops compat
        season=bundle["fiscal_years"],  # hoops compat
        archetype=bundle["archetypes"],
    )
    (out_path / "feature_manifest.json").write_text(
        json.dumps(bundle["feature_manifest"], indent=2)
    )
    print(f"Saved {len(bundle['Z'])} rows x {bundle['Z'].shape[1]} feats to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=int, default=800)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--out", type=str, default="pipeline/data")
    args = ap.parse_args()
    bundle = gen_company_profile(n_companies=args.companies, n_years=args.years)
    save_bundle(bundle, Path(args.out))
