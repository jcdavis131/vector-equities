# Towers v6 package — 20 families (17→20) PRODUCTION-ONLY NO SYNTHETIC per 2026-08-15 directive
# Free platform, no pip, torch auto cuda else cpu, ACNE optional local
"""
v6 17→20 upgrade: 10 industry_event +10 political_risk +12 global_trade_commodity =32 feats
118 →150 (or 122→154 with alt base). Train matrix 14400×154 6.8MB npz fine.
Zero-deps true: stdlib + numpy only PRODUCTION-ONLY per 2026-08-15. GDELT/yfinance/GPR/EPU/GSCPI
real fetch required, NO synthetic fallback — honest 503 if missing per user directive.

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
    import sys
    print("[equities v6] BLOCKED: synthetic_industry_rows removed — production-only per 2026-08-15", file=sys.stderr)
    print("[equities v6] Real GDELT/yfinance/GPR fetch required — honest 503", file=sys.stderr)
    sys.exit(2)


def synthetic_matrix(N, sectors_arr, fy_arr, seed=42):
    import sys
    print("[equities v6] BLOCKED: synthetic_matrix removed — production-only per 2026-08-15", file=sys.stderr)
    print("[equities v6] Real GDELT/yfinance/trade fetch required — honest 503", file=sys.stderr)
    sys.exit(2).astype(np.float32)


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
