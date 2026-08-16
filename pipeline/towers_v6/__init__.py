# Towers v6 package — 20 families (17→20) with synthetic fallback offline zero-deps true
# Free platform, no pip, torch auto cuda else cpu, ACNE optional local
"""
v6 17→20 upgrade: 10 industry_event +10 political_risk +12 global_trade_commodity =32 feats
118 →150 (or 122→154 with alt base). Train matrix 14400×154 6.8MB npz fine.
Zero-deps true: stdlib + numpy only for synthetic fallback. GDELT/yfinance/GPR/EPU/GSCPI
fetch optional, offline triggers synthetic proxy sector-specific noise.

Model auto-detects families via family_slices(manifest) → ResidualTower per family.
No code change needed in EquitiesMTNN ContinuousFusion / TransformerFusion —
fusion_mode auto attends over n_towers 17→20.

Config for transformer 4L4H d_model96 tower_width24 batch512 OneCycle 10% warmup clip1.0
dropout0.12 temp0.08 hard-neg0.2

Builder1 T5 hill133 lite 5m smoke, defer full 60ep to LOCAL-GPU.
"""

from __future__ import annotations

import numpy as np

INDUSTRY_FEATURES = [
    "IND_NEWS_VOL_Z",
    "IND_NEWS_TONE_AVG",
    "IND_NEG_EVENT_CNT",
    "IND_POS_EVENT_CNT",
    "IND_REGULATORY_RISK",
    "IND_MA_INTENSITY",
    "IND_SUPPLY_DISRUPTION",
    "IND_EARN_BREADTH",
    "IND_DISPERSION_MOM",
    "IND_VOL_SPIKE",
]

POLITICAL_FEATURES = [
    "GPR_GLOBAL_AVG_FY",
    "GPR_YOY",
    "EPU_US_AVG_FY",
    "EPU_GLOBAL_AVG_FY",
    "ELEC_PROX_US",
    "ELEC_PROX_GLOBAL",
    "TARIFF_RISK",
    "WGI_POL_STABILITY",
    "GOV_SHUTDOWN_PROX",
    "RATE_VOL_3M",
]

TRADE_FEATURES = [
    "OIL_WTI_YOY",
    "OIL_BRENT_SPREAD",
    "COPPER_YOY",
    "STEEL_PROXY_YOY",
    "LUMBER_YOY",
    "NATGAS_YOY",
    "DXY_YOY",
    "USDCNY_YOY",
    "BDRY_YOY",
    "GSCPI_AVG_FY",
    "COMMODITY_BETA_X_SECTOR",
    "AGRI_YOY",
]

NEW_FEATURES = INDUSTRY_FEATURES + POLITICAL_FEATURES + TRADE_FEATURES
NEW_FAMILIES = (
    ["industry_event"] * len(INDUSTRY_FEATURES)
    + ["political_risk"] * len(POLITICAL_FEATURES)
    + ["global_trade_commodity"] * len(TRADE_FEATURES)
)

assert len(NEW_FEATURES) == 32
assert len(NEW_FAMILIES) == 32


# Synthetic generators — zero-deps numpy only, offline safe
def _rng(seed=42):
    return np.random.default_rng(seed)


def synthetic_industry_rows(n=14400, sectors=None, years=None, seed=42):
    _rng(seed)
    # Return dict sector->year->vector but for matrix generator we generate per-row later
    return None  # placeholder for pandas path


def synthetic_matrix(N, sectors_arr, fy_arr, seed=42):
    """
    Generate Z_new N×32 synthetic that respects sector/year noise.
    Sectors: array of str len N, fy: array int or str like 2015 etc.
    Zero-deps, offline, deterministic seed.
    Mirrors logic in industry_gdelt.pol synthetic + political + trade
    but numpy-only.
    """
    rng = np.random.default_rng(seed)
    Z_new = np.zeros((N, 32), dtype=np.float32)

    # sector sensitivity for commodity beta
    sens_map = {
        "Energy": 1.5,
        "Materials": 1.2,
        "Industrials": 0.8,
        "Consumer Discretionary": 0.3,
        "Consumer Staples": 0.1,
        "Healthcare": 0.0,
        "Financials": 0.2,
        "Technology": 0.1,
        "Communication": 0.1,
        "Utilities": 0.4,
        "Real Estate": 0.2,
        # underscore variants in manifest
        "Consumer_Discretionary": 0.3,
        "Consumer_Staples": 0.1,
        "Real_Estate": 0.2,
    }

    for i in range(N):
        sec = str(sectors_arr[i]) if sectors_arr is not None else "Technology"
        fy = int(str(fy_arr[i])[:4]) if fy_arr is not None else 2020

        # INDUSTRY 10
        # base per sector/year noise correlated
        base_vol = rng.normal(0, 0.5) + 0.5 * np.sin(fy * 0.5) * 0.3
        tone = rng.normal(0, 1.0) + (0.2 if "Tech" in sec else -0.1 if "Energy" in sec else 0)
        neg = rng.poisson(3 + (2 if sec in ["Energy", "Materials"] else 0))
        pos = rng.poisson(4 + (1 if "Tech" in sec else 0))
        reg = rng.poisson(2 + (2 if sec in ["Financials", "Healthcare", "Technology"] else 0))
        ma = rng.poisson(3)
        supply = rng.poisson(2 + (3 if fy >= 2020 else 0))
        earn_breadth = rng.normal(0.5, 0.15)
        mom_disp = rng.uniform(0.1, 0.4)
        vol_spike = rng.uniform(0.8, 1.5) + (0.3 if fy in [2020, 2022] else 0)
        # z-ish
        Z_new[i, 0] = (base_vol - 0.5) / 0.6
        Z_new[i, 1] = tone / 1.5
        Z_new[i, 2] = (neg - 3) / 2.0
        Z_new[i, 3] = (pos - 4) / 2.0
        Z_new[i, 4] = (reg - 2) / 1.5
        Z_new[i, 5] = (ma - 3) / 1.8
        Z_new[i, 6] = (supply - 2) / 2.0
        Z_new[i, 7] = (earn_breadth - 0.5) / 0.15
        Z_new[i, 8] = (mom_disp - 0.25) / 0.1
        Z_new[i, 9] = (vol_spike - 1.0) / 0.3

        # POLITICAL 10 — market-wide FY conditioning
        gpr = 100 + (50 if fy == 2022 else 20 if fy in [2020, 2024] else 0) + rng.normal(0, 10)
        gpr_yoy = gpr - (100 + rng.normal(0, 5))
        epu_us = 100 + (80 if fy in [2020, 2016, 2024] else 0) + rng.normal(0, 15)
        epu_global = epu_us * 0.8 + rng.normal(0, 10)
        tariff = 50 + (100 if fy in [2018, 2019, 2024, 2025] else 0) + rng.normal(0, 10)
        elec_us = 1.0 if fy in [2016, 2020, 2024] else 0.7 if fy in [2015, 2019, 2023] else 0.2
        elec_global = rng.uniform(0.3, 0.8) + (0.2 if fy % 2 == 0 else 0)
        wgi = rng.normal(0.5, 0.1)
        gov_shutdown = 1.0 if fy in [2018, 2023] else rng.uniform(0, 0.3)
        rate_vol = 0.5 + (1.5 if fy in [2022, 2023] else 0.2) + rng.normal(0, 0.2)
        # z-score-ish already handled by subtracting mean/std later via simple scaling
        # we pre-z via historic approx means
        Z_new[i, 10] = (gpr - 110) / 15
        Z_new[i, 11] = gpr_yoy / 10
        Z_new[i, 12] = (epu_us - 110) / 20
        Z_new[i, 13] = (epu_global - 95) / 18
        Z_new[i, 14] = elec_us - 0.5
        Z_new[i, 15] = elec_global - 0.5
        Z_new[i, 16] = (tariff - 60) / 30
        Z_new[i, 17] = (wgi - 0.5) / 0.1
        Z_new[i, 18] = gov_shutdown - 0.3
        Z_new[i, 19] = (rate_vol - 0.7) / 0.4

        # TRADE 12
        oil_yoy = rng.normal(0.1, 0.4) + (-0.5 if fy == 2020 else 0.6 if fy == 2022 else 0)
        brent_spread = rng.normal(2, 1)
        copper_yoy = rng.normal(0.05, 0.3) + (0.4 if fy == 2021 else -0.2 if fy == 2020 else 0)
        steel_yoy = rng.normal(0.08, 0.35)
        lumber_yoy = rng.normal(0.0, 0.5) + (1.0 if fy == 2021 else -0.6 if fy == 2022 else 0)
        natgas_yoy = rng.normal(0.2, 0.6) + (1.2 if fy == 2022 else 0)
        dxy_yoy = rng.normal(0.02, 0.08) + (0.1 if fy == 2022 else -0.05 if fy == 2023 else 0)
        usdcny_yoy = rng.normal(0.02, 0.06) + (0.08 if fy in [2018, 2019] else 0)
        bdry_yoy = rng.normal(0.1, 0.8) + (-0.6 if fy == 2020 else 1.0 if fy == 2021 else 0)
        gscpi = rng.normal(0, 1) + (2.5 if fy in [2021, 2022] else -0.5 if fy == 2023 else 0)
        sens = sens_map.get(sec, 0.5)
        commodity_beta = copper_yoy * sens
        agri_yoy = rng.normal(0.05, 0.3) + (0.5 if fy == 2022 else 0)

        Z_new[i, 20] = oil_yoy / 0.4
        Z_new[i, 21] = (brent_spread - 2) / 1.0
        Z_new[i, 22] = copper_yoy / 0.3
        Z_new[i, 23] = steel_yoy / 0.35
        Z_new[i, 24] = lumber_yoy / 0.5
        Z_new[i, 25] = natgas_yoy / 0.6
        Z_new[i, 26] = dxy_yoy / 0.08
        Z_new[i, 27] = usdcny_yoy / 0.06
        Z_new[i, 28] = bdry_yoy / 0.8
        Z_new[i, 29] = gscpi
        Z_new[i, 30] = commodity_beta / 0.5
        Z_new[i, 31] = agri_yoy / 0.3

    return Z_new.astype(np.float32)


def get_v6_features():
    return NEW_FEATURES.copy(), NEW_FAMILIES.copy()


def get_manifest_extension():
    return {
        "features": NEW_FEATURES,
        "families": NEW_FAMILIES,
        "n_new": 32,
        "families_17_to_20": ["industry_event", "political_risk", "global_trade_commodity"],
        "config": {
            "tower_width": 24,
            "d_model": 96,
            "n_fusion_layers": 4,
            "n_attn_heads": 4,
            "batch": 512,
            "one_cycle": True,
            "pct_start": 0.1,
            "clip": 1.0,
            "dropout": 0.12,
            "temp": 0.08,
            "hard_neg": 0.2,
            "lr": 1.5e-3,
            "weight_decay": 1e-4,
        },
    }
