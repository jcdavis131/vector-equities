import json, sys
sys.path.insert(0,"pipeline")
from pathlib import Path
import numpy as np, pandas as pd
from dataset_career import load_bundle

DATA_DIR = Path("pipeline/data")
npz = np.load(DATA_DIR/"train_matrix_v5.npz", allow_pickle=True)
Z = npz["Z"]
mask = npz["mask"]
print(f"Loaded v5 Z {Z.shape}")

manifest = json.loads((DATA_DIR/"feature_manifest_v5.json").read_text())

from towers_v6.industry_gdelt import synthetic_industry_features
from towers_v6.political import synthetic_political
from towers_v6.trade_commodities import synthetic_trade

ind_df = synthetic_industry_features()
pol_df = synthetic_political()
trade_df = synthetic_trade()

Z2, mask2, Z_raw, tickers, names, fy_arr, sectors_arr, manifest2, fwd, path = load_bundle()
N = Z.shape[0]

ind_lookup = {(row.sector, row.year): row for row in ind_df.itertuples()}
pol_lookup = {row.year: row for row in pol_df.itertuples()}
trade_lookup = {row.year: row for row in trade_df.itertuples()}

ind_cols = [c for c in ind_df.columns if c not in ["sector","year"]]
pol_cols = [c for c in pol_df.columns if c!="year"]
trade_cols = [c for c in trade_df.columns if c!="year"]

new_features = ind_cols + pol_cols + trade_cols
new_families = ["industry_event"]*len(ind_cols) + ["political_risk"]*len(pol_cols) + ["global_trade_commodity"]*len(trade_cols)

print(f"New features {len(new_features)}")

D_new = len(new_features)
Z_new = np.zeros((N, D_new), dtype=np.float32)
mask_new = np.ones((N, D_new), dtype=np.float32)

sens_map = {"Energy":1.5,"Materials":1.2,"Industrials":0.8,"Consumer Discretionary":0.3,"Consumer Staples":0.1,"Healthcare":0.0,"Financials":0.2,"Technology":0.1,"Communication":0.1,"Utilities":0.4,"Real Estate":0.2}

for i in range(N):
    fy = int(fy_arr[i])
    sector = sectors_arr[i]
    # industry
    key = (sector, fy)
    if key in ind_lookup:
        row = ind_lookup[key]
        for j,col in enumerate(ind_cols):
            Z_new[i,j] = float(getattr(row, col))
    else:
        # use year mean
        fallback = ind_df[ind_df.year==fy]
        if len(fallback)>0:
            for j,col in enumerate(ind_cols):
                Z_new[i,j] = float(fallback[col].mean())
    # political
    if fy in pol_lookup:
        prow = pol_lookup[fy]
        for j2,col in enumerate(pol_cols):
            idx = len(ind_cols)+j2
            Z_new[i, idx] = float(getattr(prow, col))
    # trade
    if fy in trade_lookup:
        trow = trade_lookup[fy]
        for j3,col in enumerate(trade_cols):
            idx = len(ind_cols)+len(pol_cols)+j3
            if col=="COMMODITY_BETA_X_SECTOR":
                sens = sens_map.get(sector,0.5)
                Z_new[i, idx] = float(getattr(trow, "COPPER_YOY")) * sens
            else:
                Z_new[i, idx] = float(getattr(trow, col))

Z_v6 = np.concatenate([Z, Z_new], axis=1)
mask_v6 = np.concatenate([mask, mask_new], axis=1)
print(f"Z_v6 {Z_v6.shape}")

manifest_v6 = {
    "features": manifest["features"] + new_features,
    "families": manifest["families"] + new_families,
    "game_features": manifest.get("game_features", []),
    "sectors": manifest.get("sectors", []),
    "real_data": manifest.get("real_data", {}),
    "v5": True, "v6": True,
    "years": manifest.get("years", []),
    "tickers": manifest.get("tickers", []),
    "rows": manifest.get("rows", 0),
    "real_flags": manifest.get("real_flags", {}),
    "real_coverage": manifest.get("real_coverage", {}),
    "sources": manifest.get("sources", {}) | {"v6_towers": ["GDELT (public)","GPR (academic)","EPU (academic)","GSCPI (NY Fed)","yfinance commodities"]},
    "forward_labels": manifest.get("forward_labels", [])
}

out_path = DATA_DIR/"train_matrix_v6.npz"
npz_v5 = dict(np.load(DATA_DIR/"train_matrix_v5.npz", allow_pickle=True))
npz_v5["Z"] = Z_v6
npz_v5["mask"] = mask_v6
np.savez_compressed(out_path, **npz_v5)
print(f"Saved {out_path}")

(Path(DATA_DIR)/"feature_manifest_v6.json").write_text(json.dumps(manifest_v6, indent=2))
print(f"Saved manifest v6 {len(manifest_v6['features'])}")

from collections import Counter
print(Counter(manifest_v6["families"]))
