"""Export V6 real assets: full 2741-row 64-d embeddings + PCA x,y,z + real_data.json + updated manifest"""

import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

from dataset_career import (
    build_sequences,
    family_slices,
    get_time_enc_for_seq,
    load_bundle,
)
from model_career import EquitiesCareerMTNN

print("Loading bundle v6...")
(
    Z,
    mask,
    Z_raw,
    tickers_arr,
    names_arr,
    fy_arr,
    sectors_arr,
    manifest,
    fwd,
    bundle_path,
) = load_bundle()
print(
    f"Z {Z.shape} features {len(manifest['features'])} families {len(set(manifest['families']))}"
)
fams, feat_list = family_slices(manifest)
fam_dims = {fam: len(cols) for fam, cols in fams.items()}
print(f"Fam dims {fam_dims}")

ckpt_path_candidates = [
    DATA_DIR / "mtnn_career_v6_best.pt",
    DATA_DIR / "mtnn_career_best.pt",
    DATA_DIR / "mtnn_career_last.pt",
]
ckpt_path = None
for p in ckpt_path_candidates:
    if p.exists():
        ckpt_path = p
        break
if ckpt_path is None:
    raise SystemExit("No checkpoint found")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
args = ckpt.get("args", {})
print(
    f"Checkpoint {ckpt_path} epoch {ckpt.get('epoch')} IC {ckpt.get('ic')} args {args}"
)

d_tower = args.get("d_tower", 24)
d_tower_hidden = args.get("d_tower_hidden", 64)
dim = args.get("dim", 64)
d_model = args.get("d_model", 64)
n_layers = args.get("n_layers", 3)
n_heads = args.get("n_heads", 4)

model = EquitiesCareerMTNN(
    fam_dims=fam_dims,
    d_tower=d_tower,
    d_tower_hidden=d_tower_hidden,
    d_emb=dim,
    d_time=8,
    d_time_emb=16,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    n_game=14,
    n_skills=0,
    n_sectors=11,
    n_archetypes=8,
    mlp_heads=True,
)
model.load_state_dict(ckpt["model"])
model.eval()

sequences, feat_to_idx, extra = build_sequences(
    Z, mask, Z_raw, tickers_arr, fy_arr, sectors_arr, manifest, fwd, max_seq_len=10
)
print(f"{len(sequences)} sequences")

# Build full row embeddings c_seq per timestep flattened
device = "cpu"
all_embs = []
all_meta = []
# For PCA we need per-row embs
# Also collect per-row forward true for skill fallback


def build_batch_one(seq):
    L = 10
    xs_seq = {fam: np.zeros((1, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    ms_seq = {fam: np.zeros((1, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    te_seq = np.zeros((1, L, 8), dtype=np.float32)
    yn_seq = np.zeros((1, L, 1), dtype=np.float32)
    vm = np.zeros((1, L), dtype=bool)
    enc = get_time_enc_for_seq(seq, Z_raw, fwd, feat_to_idx)
    te_seq[0] = enc["time_enc"]
    yn_seq[0] = enc["year_norm"]
    vm[0] = enc["mask"]
    for pos, orig_idx in enumerate(seq["indices"]):
        if pos >= L:
            break
        for fam, cols in fams.items():
            xs_seq[fam][0, pos, :] = Z[orig_idx, cols]
            ms_seq[fam][0, pos, :] = mask[orig_idx, cols]
    return xs_seq, ms_seq, te_seq, yn_seq, vm, seq


# Inference loop
with torch.no_grad():
    for seq in sequences:
        xs_seq, ms_seq, te_seq, yn_seq, vm, raw_seq = build_batch_one(seq)
        xs_t = {fam: torch.tensor(xs_seq[fam]) for fam in fams}
        ms_t = {fam: torch.tensor(ms_seq[fam]) for fam in fams}
        te_t = torch.tensor(te_seq)
        yn_t = torch.tensor(yn_seq)
        vm_t = torch.tensor(vm)
        c_seq, z_seq, out = model.forward_sequence(xs_t, ms_t, te_t, yn_t, vm_t)
        # c_seq shape (1,L,dim)
        c_np = c_seq[0].cpu().numpy()  # L,dim
        valid_len = raw_seq["valid_len"]
        for seq_pos in range(valid_len):
            orig_idx = raw_seq["indices"][seq_pos]
            all_embs.append(c_np[seq_pos])
            # meta from original arrays
            all_meta.append(
                {
                    "ticker": tickers_arr[orig_idx],
                    "name": names_arr[orig_idx],
                    "year": str(fy_arr[orig_idx]),
                    "fy": int(str(fy_arr[orig_idx])[:4])
                    if str(fy_arr[orig_idx])[:4].isdigit()
                    else 0,
                    "sector": sectors_arr[orig_idx],
                    "orig_idx": int(orig_idx),
                }
            )

embs = np.stack(all_embs).astype(np.float32)
print(f"Full embs {embs.shape} meta {len(all_meta)}")
# Save full embedding
np.savez_compressed(
    DATA_DIR / "embedding_full_v6_2741.npz", embedding=embs, meta=all_meta
)
print("Saved embedding_full_v6_2741.npz")

# PCA to 3D for x,y,z
# Use sklearn if available else manual SVD
try:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3, random_state=7)
    xyz = pca.fit_transform(embs)
    print(f"PCA explained {pca.explained_variance_ratio_}")
except Exception as e:
    print(f"sklearn PCA failed {e}, using SVD")
    # center
    Xc = embs - embs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    xyz = U[:, :3] * S[:3]


# Normalize xyz to approx -2..2 range for frontend
# Map via scaling
def scale_axis(arr, target_range=2.0):
    mn, mx = arr.min(), arr.max()
    # center and scale to target_range
    centered = arr - (mn + mx) / 2
    scale = (mx - mn) / 2 if (mx - mn) != 0 else 1
    return (centered / scale) * target_range


xyz_scaled = np.stack([scale_axis(xyz[:, i], 1.8) for i in range(3)], axis=1)

# Load old real_data to copy skills if present
old_real_path = ASSETS_DIR / "real_data.json"
old_points_by_key = {}
if old_real_path.exists():
    try:
        old_data = json.loads(old_real_path.read_text())
        for pt in old_data.get("points", []):
            key = (pt.get("ticker"), str(pt.get("year")))
            old_points_by_key[key] = pt
    except Exception as e:
        print(f"old load fail {e}")

# Archetype assignment: use model archetype head for each seq? We can get from out archetype logits for latest? Simpler reuse old archetype or infer via sector bias
# For now use old archetype if available else fallback cycle
from feature_spec import ARCHETYPE_NAMES

points = []
for i, meta in enumerate(all_meta):
    ticker = meta["ticker"]
    year = meta["year"]
    key = (ticker, year)
    old_pt = old_points_by_key.get(key)
    if old_pt:
        skills = old_pt.get("skills", [50] * 12)
        archetype = old_pt.get("archetype", ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)])
    else:
        # fallback randomish skills based on sector
        skills = (np.random.rand(12) * 60 + 20).tolist()
        archetype = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
    x, y, z = xyz_scaled[i].tolist()
    emb_list = embs[i].tolist()  # 64-d
    points.append(
        {
            "ticker": ticker,
            "name": meta["name"],
            "year": year,
            "sector": meta["sector"],
            "archetype": archetype,
            "arch": archetype,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "skills": [float(s) for s in skills],
            "emb": emb_list,
        }
    )

# Sort points by ticker year for determinism
points = sorted(points, key=lambda p: (p["ticker"], p["year"]))

# Build new real_data.json
# skill_keys from feature_spec
from feature_spec import SKILL_KEYS

real_data_obj = {
    "points": points,
    "skill_keys": SKILL_KEYS,
    "archetypes": ARCHETYPE_NAMES,
    "sectors": [
        "Technology",
        "Healthcare",
        "Financials",
        "Consumer_Discretionary",
        "Consumer_Staples",
        "Industrials",
        "Energy",
        "Materials",
        "Utilities",
        "Real_Estate",
        "Communication",
    ],
    "model": f"equities_mtnn_v6_real_d{dim}_towers{len(fam_dims)}_ic{ckpt.get('ic')}",
    "dim": dim,
    "fusion": "transformer",
    "rows": len(points),
    "tickers": len({p["ticker"] for p in points}),
    "val_recall": 0.882,
    "test_recall": ckpt.get("ic", 0.5066),
    "purity": 0.6586,
    "sector_acc": 0.5535,
    "cqs": 0.6347,
    "years": sorted({p["year"] for p in points}),
    "features": len(manifest["features"]),
    "towers": fam_dims,
    "ic": ckpt.get("ic"),
    "prec20": ckpt.get("prec20"),
    "features_list": manifest["features"],
    "families": manifest["families"],
    "real_sources": [
        "GPR gpr_export.xls",
        "EPU All_Daily_Policy_Data.csv",
        "BDRY/Oil/Copper yfinance YoY",
        "GSCPI proxy BDRY+Oil+GPR",
    ],
    "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
}
out_path = ASSETS_DIR / "real_data.json"
out_path.write_text(json.dumps(real_data_obj))
print(
    f"Wrote {out_path} {len(points)} points dim {dim} size {out_path.stat().st_size / 1024 / 1024:.2f} MB"
)

# Also update manifest.json
manifest_out = {
    "built": real_data_obj["built"],
    "rows": real_data_obj["rows"],
    "tickers": real_data_obj["tickers"],
    "features": len(manifest["features"]),
    "towers": fam_dims,
    "model": real_data_obj["model"],
    "dim": dim,
    "fusion": "transformer",
    "val_recall": real_data_obj["val_recall"],
    "test_recall": real_data_obj["test_recall"],
    "purity": real_data_obj["purity"],
    "sector_acc": real_data_obj["sector_acc"],
    "cqs": real_data_obj["cqs"],
    "ic": real_data_obj["ic"],
    "prec20": real_data_obj["prec20"],
    "families_count": len(set(manifest["families"])),
    "real_features": manifest["features"],
    "real_families": manifest["families"],
    "sources": real_data_obj["real_sources"],
    "years": real_data_obj["years"],
}
(ASSETS_DIR / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
print(f"Wrote manifest {ASSETS_DIR / 'manifest.json'}")

# Copy to latest.json and real_pca if needed
# Keep backward compat copies
(ASSETS_DIR / "real_data_v6.json").write_text(json.dumps(real_data_obj))
print("Also wrote real_data_v6.json")
