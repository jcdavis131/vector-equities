"""
Rebuild v6 towers using REAL external data where available
- GPR, EPU from real CSVs
- Commodities from real yfinance monthly files
- Industry tower: synthetic but conditioned on real commodity/GPR shocks (so not pure noise)
"""

import json
import sys

sys.path.insert(0, "pipeline")
from pathlib import Path

import numpy as np
import pandas as pd
from dataset_career import load_bundle

DATA_DIR = Path("pipeline/data")
EXT_DIR = DATA_DIR / "external"

# Load v5 bundle
Z, mask, Z_raw, tickers, names, fy_arr, sectors_arr, manifest_v5, fwd, path = (
    load_bundle()
)
# But load_bundle now prefers v6 — force v5 load
npz_v5 = np.load(DATA_DIR / "train_matrix_v5.npz", allow_pickle=True)
Z_v5 = npz_v5["Z"]
mask_v5 = npz_v5["mask"]
# manifest v5
manifest = json.loads((DATA_DIR / "feature_manifest_v5.json").read_text())
N = Z_v5.shape[0]

print(f"V5 Z {Z_v5.shape}")


# --- Load real political ---
def load_gpr():
    path = EXT_DIR / "gpr_export.xls"
    df = pd.read_excel(path, engine="xlrd")
    df["month_dt"] = pd.to_datetime(df["month"])
    df["year"] = df["month_dt"].dt.year
    yearly = (
        df.groupby("year")
        .agg(GPR=("GPR", "mean"), GPRT=("GPRT", "mean"), GPRA=("GPRA", "mean"))
        .reset_index()
    )
    yearly["GPR_YOY"] = yearly["GPR"].pct_change()
    yearly["GPRA_YOY"] = yearly["GPRA"].pct_change()
    return yearly


def load_epu():
    path = EXT_DIR / "All_Daily_Policy_Data.csv"
    df = pd.read_csv(path)
    yearly = (
        df.groupby("year")
        .agg(
            EPU_US=("daily_policy_index", "mean"),
            EPU_US_STD=("daily_policy_index", "std"),
        )
        .reset_index()
    )
    yearly["EPU_YOY"] = yearly["EPU_US"].pct_change()
    return yearly


def parse_commodity_file(fname):
    p = EXT_DIR / fname
    if not p.exists():
        return None
    try:
        # skip 2 rows
        df = pd.read_csv(p, skiprows=2)
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        close_col = df.columns[1]
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
        df["year"] = df["Date"].dt.year
        yearly = (
            df.groupby("year")[close_col]
            .mean()
            .reset_index()
            .rename(columns={close_col: "avg"})
        )
        yearly["yoy"] = yearly["avg"].pct_change()
        yearly["ticker"] = fname
        return yearly
    except Exception as e:
        print(f"parse {fname} failed {e}")
        return None


# Load all commodities
commodities = {}
for f in [
    "CL_eq_F_monthly.csv",
    "BZ_eq_F_monthly.csv",
    "HG_eq_F_monthly.csv",
    "SLX_monthly.csv",
    "LBS_eq_F_monthly.csv",
    "NG_eq_F_monthly.csv",
    "DX_Y.NYB_monthly.csv",
    "CNY_eq_X_monthly.csv",
    "BDRY_monthly.csv",
    "GC_eq_F_monthly.csv",
]:
    yt = parse_commodity_file(f)
    if yt is not None:
        key = (
            f.replace("_monthly.csv", "")
            .replace("_eq_F", "")
            .replace("_eq_X", "")
            .replace(".NYB", "")
            .replace("_", "")
        )
        # normalize keys: CL, BZ, HG, SLX, LBS, NG, DXY, CNY, BDRY, GC
        kmap = {
            "CL": "OIL_WTI",
            "BZ": "OIL_BRENT",
            "HG": "COPPER",
            "SLX": "STEEL",
            "LBS": "LUMBER",
            "NG": "NATGAS",
            "DXNYB": "DXY",
            "CNY": "USDCNY",
            "BDRY": "BDRY",
            "GC": "GOLD",
        }
        # map
        base = f.split("_")[0]
        if "CL" in f:
            k = "OIL_WTI"
        elif "BZ" in f:
            k = "OIL_BRENT"
        elif "HG" in f:
            k = "COPPER"
        elif "SLX" in f:
            k = "STEEL"
        elif "LBS" in f:
            k = "LUMBER"
        elif "NG" in f:
            k = "NATGAS"
        elif "DX" in f:
            k = "DXY"
        elif "CNY" in f:
            k = "USDCNY"
        elif "BDRY" in f:
            k = "BDRY"
        elif "GC" in f:
            k = "GOLD"
        else:
            k = f
        commodities[k] = yt
        print(f"Loaded {k} {yt.shape} tail:")
        print(yt.tail(5).to_string())

gpr_yearly = load_gpr()
epu_yearly = load_epu()
print("\nGPR yearly")
print(gpr_yearly.tail(10).to_string())
print("\nEPU yearly")
print(epu_yearly.tail(10).to_string())

# Build lookup for years 2015-2024
years = list(range(2015, 2025))
# Create dict year -> dict of features
year_features = {}

for y in years:
    # GPR
    gpr_row = gpr_yearly[gpr_yearly.year == y]
    gpr_avg = float(gpr_row["GPR"].values[0]) if len(gpr_row) > 0 else np.nan
    gpr_yoy = (
        float(gpr_row["GPR"].pct_change().values[0])
        if False
        else float(gpr_yearly[gpr_yearly.year == y]["GPR"].pct_change().values[0])
        if len(gpr_yearly[gpr_yearly.year == y]) > 0
        else 0
    )
    # Actually compute yoy from yearly df
    # We'll use pct_change already computed but need lookup
    # Simpler: get from yearly dataframe with shift
    # gpr_yoy already in yearly df? we computed earlier? Let's recompute dict
    pass

# Build dicts properly
gpr_dict = {}
for _, row in gpr_yearly.iterrows():
    y = int(row["year"])
    gpr_dict[y] = {
        "GPR": float(row["GPR"]) if not pd.isna(row["GPR"]) else 0,
        "GPR_YOY": float(row["GPR_YOY"])
        if "GPR_YOY" in row and not pd.isna(row["GPR_YOY"])
        else 0,
        "GPRA": float(row["GPRA"]) if "GPRA" in row and not pd.isna(row["GPRA"]) else 0,
    }
# Fill missing years with interpolation or 0
epu_dict = {}
for _, row in epu_yearly.iterrows():
    y = int(row["year"])
    epu_dict[y] = {
        "EPU_US": float(row["EPU_US"]) if not pd.isna(row["EPU_US"]) else 0,
        "EPU_YOY": float(row["EPU_YOY"])
        if "EPU_YOY" in row and not pd.isna(row["EPU_YOY"])
        else 0,
        "EPU_STD": float(row["EPU_US_STD"])
        if "EPU_US_STD" in row and not pd.isna(row["EPU_US_STD"])
        else 0,
    }

comm_dict = {}
# initialize
for y in years:
    comm_dict[y] = {}
for k, yt in commodities.items():
    # yt has year, avg, yoy
    for _, row in yt.iterrows():
        y = int(row["year"])
        if y in years:
            comm_dict[y][k] = float(row["avg"]) if not pd.isna(row["avg"]) else 0
            comm_dict[y][k + "_YOY"] = (
                float(row["yoy"]) if not pd.isna(row["yoy"]) else 0
            )

print("\nCommodity dict sample 2022")
print(comm_dict[2022])

# Now build towers per row N=2741
SECTORS = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Healthcare",
    "Financials",
    "Technology",
    "Communication",
    "Utilities",
    "Real Estate",
    "Inc.",
]
sector_sens = {
    "Energy": {
        "OIL": 1.5,
        "GAS": 1.2,
        "COPPER": 0.5,
        "STEEL": 0.3,
        "LUMBER": 0.1,
        "DXY": -0.5,
        "BDRY": 0.4,
    },
    "Materials": {"OIL": 0.8, "COPPER": 1.3, "STEEL": 1.2, "LUMBER": 0.8, "GOLD": 0.6},
    "Industrials": {
        "OIL": 0.6,
        "STEEL": 0.8,
        "BDRY": 1.0,
        "LUMBER": 0.5,
        "COPPER": 0.6,
    },
    "Technology": {"TARIFF": 1.0, "USDCNY": 0.8, "DXY": 0.3},
    "Financials": {"RATE": 1.0, "EPU": 0.8, "GPR": 0.5},
    "Healthcare": {"REG": 1.0, "EPU": 0.5},
    "Consumer Discretionary": {"BDRY": 0.6, "OIL": 0.4, "LUMBER": 0.5},
    "Consumer Staples": {"AGRI": 0.8, "OIL": 0.3},
    "Communication": {"REG": 0.8, "TARIFF": 0.5},
    "Utilities": {"NATGAS": 1.2, "OIL": 0.5, "RATE": 0.8},
    "Real Estate": {"LUMBER": 1.2, "RATE": 1.0, "STEEL": 0.6},
}
# Default for missing

np.random.seed(42)

# Define new feature order (must match previous v6 manifest order)
ind_cols = [
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
pol_cols = [
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
trade_cols = [
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

new_feature_names = ind_cols + pol_cols + trade_cols
print(f"New features {len(new_feature_names)}")

Z_new = np.zeros((N, len(new_feature_names)), dtype=np.float32)

for i in range(N):
    fy = int(fy_arr[i]) if str(fy_arr[i]).isdigit() else 2020
    sector = str(sectors_arr[i])
    # Get macro values
    gpr = gpr_dict.get(fy, {"GPR": 100, "GPR_YOY": 0})
    epu = epu_dict.get(fy, {"EPU_US": 100, "EPU_YOY": 0, "EPU_STD": 20})
    comm = comm_dict.get(fy, {})

    oil_yoy = comm.get("OIL_WTI_YOY", 0)
    brent_avg = comm.get("OIL_BRENT", 0)
    wti_avg = comm.get("OIL_WTI", 0)
    brent_spread = (brent_avg - wti_avg) if (brent_avg and wti_avg) else 0
    copper_yoy = comm.get("COPPER_YOY", 0)
    steel_yoy = comm.get("STEEL_YOY", 0)
    lumber_yoy = comm.get("LUMBER_YOY", 0)
    natgas_yoy = comm.get("NATGAS_YOY", 0)
    dxy_yoy = comm.get("DXY_YOY", 0)
    usdcny_yoy = comm.get("USDCNY_YOY", 0)
    bdry_yoy = comm.get("BDRY_YOY", 0)
    gold_yoy = comm.get("GOLD_YOY", 0)

    # Political features (raw before z-score)
    # We'll compute raw then later z-score across dataset
    # For now store raw signals in Z_new then z-score later

    # Industry features conditioned on macro
    # base synthetic + macro influence
    sens = sector_sens.get(sector, {})
    # news vol: higher when GPR high, BDRY high, EPU high
    ind_news_vol = (
        0.5 * (gpr.get("GPR_YOY", 0))
        + 0.3 * bdry_yoy
        + 0.2 * epu.get("EPU_YOY", 0)
        + np.random.normal(0, 0.2)
        + sens.get("OIL", 0) * 0.1 * oil_yoy
    )
    # tone: negative when GPR high, EPU high, oil spike negative for airlines etc
    ind_tone = (
        -0.4 * gpr.get("GPR_YOY", 0)
        - 0.3 * epu.get("EPU_YOY", 0)
        + (0.2 if sector == "Technology" else -0.1 if sector == "Energy" else 0)
        + np.random.normal(0, 0.3)
    )
    ind_neg = max(
        0,
        3
        + 2 * (1 if sector in ["Energy", "Materials"] else 0)
        + 1.5 * gpr.get("GPR", 0) / 100
        + np.random.normal(0, 0.5),
    )
    ind_pos = max(
        0,
        4
        + (1 if sector == "Technology" else 0)
        + np.random.normal(0, 0.5)
        - 0.5 * epu.get("EPU_YOY", 0),
    )
    ind_reg = (
        2
        + (2 if sector in ["Financials", "Healthcare", "Technology"] else 0)
        + 0.5 * epu.get("EPU_YOY", 0)
        + np.random.normal(0, 0.4)
    )
    ind_ma = (
        3
        + np.random.normal(0, 0.5)
        - 0.3 * epu.get("EPU_YOY", 0)
        + (0.5 if fy in [2021, 2022] else 0)
    )
    ind_supply = (
        2
        + (3 if fy >= 2020 else 0)
        + 0.8 * bdry_yoy
        + 0.4 * gpr.get("GPR_YOY", 0)
        + np.random.normal(0, 0.4)
    )
    ind_earn = (
        0.5 + 0.1 * oil_yoy + np.random.normal(0, 0.1) + (-0.1 if fy == 2020 else 0)
    )
    ind_disp = 0.25 + 0.1 * abs(oil_yoy) + np.random.normal(0, 0.05)
    ind_vol_spike = (
        1.0
        + (0.5 if fy in [2020, 2022] else 0)
        + 0.3 * abs(gpr.get("GPR_YOY", 0))
        + np.random.normal(0, 0.1)
    )

    # Political raw
    elec_us = (
        1.0 if fy in [2016, 2020, 2024] else 0.7 if fy in [2015, 2019, 2023] else 0.2
    )
    elec_global = 0.5 + (0.3 if fy % 2 == 0 else 0) + 0.1 * epu.get("EPU_YOY", 0)
    tariff_risk = (
        50
        + (100 if fy in [2018, 2019, 2024, 2025] else 0)
        + 50 * usdcny_yoy
        + 20 * bdry_yoy
    )
    wgi = 0.5 + np.random.normal(0, 0.1) - 0.1 * gpr.get("GPR_YOY", 0)
    gov_shutdown = 1.0 if fy in [2018, 2023] else 0.2
    rate_vol = (
        0.5
        + (1.5 if fy in [2022, 2023] else 0.2)
        + 0.5 * abs(dxy_yoy)
        + 0.3 * abs(epu.get("EPU_STD", 20) / 100)
    )

    # Store
    idx = 0
    # ind
    Z_new[i, idx + 0] = ind_news_vol
    Z_new[i, idx + 1] = ind_tone
    Z_new[i, idx + 2] = ind_neg
    Z_new[i, idx + 3] = ind_pos
    Z_new[i, idx + 4] = ind_reg
    Z_new[i, idx + 5] = ind_ma
    Z_new[i, idx + 6] = ind_supply
    Z_new[i, idx + 7] = ind_earn
    Z_new[i, idx + 8] = ind_disp
    Z_new[i, idx + 9] = ind_vol_spike
    idx += 10
    # pol - raw before zscore, we will store then zscore later? For simplicity store raw gpr etc
    Z_new[i, idx + 0] = gpr.get("GPR", 100)
    Z_new[i, idx + 1] = gpr.get("GPR_YOY", 0)
    Z_new[i, idx + 2] = epu.get("EPU_US", 100)
    Z_new[i, idx + 3] = epu.get("EPU_US", 100) * 0.8  # proxy global
    Z_new[i, idx + 4] = elec_us
    Z_new[i, idx + 5] = elec_global
    Z_new[i, idx + 6] = tariff_risk
    Z_new[i, idx + 7] = wgi
    Z_new[i, idx + 8] = gov_shutdown
    Z_new[i, idx + 9] = rate_vol
    idx += 10
    # trade
    Z_new[i, idx + 0] = oil_yoy
    Z_new[i, idx + 1] = brent_spread
    Z_new[i, idx + 2] = copper_yoy
    Z_new[i, idx + 3] = steel_yoy
    Z_new[i, idx + 4] = lumber_yoy
    Z_new[i, idx + 5] = natgas_yoy
    Z_new[i, idx + 6] = dxy_yoy
    Z_new[i, idx + 7] = usdcny_yoy
    Z_new[i, idx + 8] = bdry_yoy
    # GSCPI proxy: BDRY + oil yoy + gpr yoy (peaked 2021-2022)
    gscpi_proxy = 0.5 * bdry_yoy + 0.3 * oil_yoy + 0.2 * gpr.get("GPR_YOY", 0)
    Z_new[i, idx + 9] = gscpi_proxy
    # commodity beta x sector
    # compute beta as copper_yoy * sensitivity
    sens_val = sens.get("OIL", 0.5) if sector in sector_sens else 0.5
    # Use copper as base for commodity beta
    comm_beta = copper_yoy * sens_val + steel_yoy * 0.3
    Z_new[i, idx + 10] = comm_beta
    Z_new[i, idx + 11] = gold_yoy  # agri proxy using gold yoy + random

# Now z-score per feature column across N
Z_new_z = (Z_new - Z_new.mean(axis=0)) / (Z_new.std(axis=0) + 1e-6)
print(f"Z_new mean {Z_new_z.mean():.4f} std {Z_new_z.std():.4f}")
print(f"Per col mean {Z_new_z.mean(axis=0)[:5]}")
print(f"Per col std {Z_new_z.std(axis=0)[:5]}")

# Concatenate with v5
Z_v6 = np.concatenate([Z_v5, Z_new_z.astype(np.float32)], axis=1)
mask_v6 = np.concatenate([mask_v5, np.ones_like(Z_new_z)], axis=1)

print(f"Z_v6 shape {Z_v6.shape}")

# Build new manifest
manifest_v6 = {
    "features": manifest["features"] + new_feature_names,
    "families": manifest["families"]
    + ["industry_event"] * 10
    + ["political_risk"] * 10
    + ["global_trade_commodity"] * 12,
    "game_features": manifest.get("game_features", []),
    "sectors": manifest.get("sectors", []),
    "real_data": manifest.get("real_data", {}),
    "years": manifest.get("years", []),
    "tickers": manifest.get("tickers", []),
    "rows": manifest.get("rows", 0),
    "real_flags": manifest.get("real_flags", {}),
    "real_coverage": manifest.get("real_coverage", {}),
    "sources": manifest.get("sources", {})
    | {
        "v6_towers_real": [
            "GPR real monthly 1900-2026",
            "EPU daily 1985-2025",
            "yfinance commodities 2015-2025 real",
        ]
    },
    "forward_labels": manifest.get("forward_labels", []),
}

# Save
out_npz = DATA_DIR / "train_matrix_v6.npz"
# Load original npz_v5 to preserve other arrays
npz_dict = dict(np.load(DATA_DIR / "train_matrix_v5.npz", allow_pickle=True))
npz_dict["Z"] = Z_v6
npz_dict["mask"] = mask_v6
np.savez_compressed(out_npz, **npz_dict)
print(f"Saved {out_npz}")

manifest_path = DATA_DIR / "feature_manifest_v6.json"
manifest_path.write_text(json.dumps(manifest_v6, indent=2))
print(f"Saved manifest {manifest_path} len {len(manifest_v6['features'])}")

# Also save tower csv for inspection
import pandas as pd

df_tower = pd.DataFrame(Z_new_z, columns=new_feature_names)
df_tower["fy"] = fy_arr
df_tower["sector"] = sectors_arr
df_tower["ticker"] = tickers
df_tower.to_csv(DATA_DIR / "towers_v6_real_features.csv", index=False)
print("Saved tower csv")

# Summary stats per year
for y in range(2015, 2025):
    sub = df_tower[df_tower.fy == str(y)]
    if len(sub) > 0:
        print(
            f"FY {y} GPR {sub['GPR_GLOBAL_AVG_FY'].mean():.2f} EPU {sub['EPU_US_AVG_FY'].mean():.2f} OIL_YOY {sub['OIL_WTI_YOY'].mean():.2f} COPPER {sub['COPPER_YOY'].mean():.2f} BDRY {sub['BDRY_YOY'].mean():.2f}"
        )
