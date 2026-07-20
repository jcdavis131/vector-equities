import sys

sys.path.insert(0, "pipeline")
from pathlib import Path

import numpy as np
import torch
from dataset_career import (
    build_sequences,
    family_slices,
    get_time_enc_for_seq,
    load_bundle,
)
from model_career import EquitiesCareerMTNN

DATA_DIR = Path("pipeline/data")
Z, mask, Z_raw, tickers, names, fiscal_years, sectors, manifest, fwd, npz_path = (
    load_bundle()
)
fams, _ = family_slices(manifest)
fam_dims = {fam: len(cols) for fam, cols in fams.items()}

ckpt = torch.load(
    DATA_DIR / "mtnn_career_v6_best.pt", map_location="cpu", weights_only=False
)
args = ckpt["args"]
print(
    f"Loaded best epoch {ckpt.get('epoch')} IC {ckpt.get('ic')} prec20 {ckpt.get('prec20')} args {args}"
)
model = EquitiesCareerMTNN(
    fam_dims=fam_dims,
    d_tower=args.get("d_tower", 24),
    d_tower_hidden=args.get("d_tower_hidden", 64),
    d_emb=args.get("dim", 64),
    d_time=8,
    d_time_emb=16,
    d_model=args.get("d_model", 64),
    n_layers=args.get("n_layers", 3),
    n_heads=args.get("n_heads", 4),
    n_game=14,
    n_skills=0,
    n_sectors=11,
    n_archetypes=8,
    mlp_heads=True,
)
model.load_state_dict(ckpt["model"])
model.eval()

seqs, feat_to_idx, extra_idx = build_sequences(
    Z, mask, Z_raw, tickers, fiscal_years, sectors, manifest, fwd, max_seq_len=10
)
# Evaluate latest FY per ticker - use last fiscal year in each seq
latest_fy = 2024
print(f"Latest FY assumed {latest_fy}, total seqs {len(seqs)}")


def build_batch_one(seq):
    L = 10
    xs_seq = {fam: np.zeros((1, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    ms_seq = {fam: np.zeros((1, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    te_seq = np.zeros((1, L, 8), dtype=np.float32)
    yn_seq = np.zeros((1, L, 1), dtype=np.float32)
    vm = np.zeros((1, L), dtype=bool)
    sector_ids = np.zeros((1, L), dtype=np.int64)
    enc = get_time_enc_for_seq(seq, Z_raw, fwd, feat_to_idx)
    te_seq[0] = enc["time_enc"]
    yn_seq[0] = enc["year_norm"]
    vm[0] = enc["mask"]
    # sector id placeholder
    sector_ids[0, : len(seq["indices"])] = 0
    for pos, orig_idx in enumerate(seq["indices"]):
        if pos >= L:
            break
        for fam, cols in fams.items():
            xs_seq[fam][0, pos, :] = Z[orig_idx, cols]
            ms_seq[fam][0, pos, :] = mask[orig_idx, cols]
    return xs_seq, ms_seq, te_seq, yn_seq, vm, sector_ids, seq


# Filter seqs where last FY == 2024 or max FY ==2024
latest_seqs = [s for s in seqs if s["fiscal_years"][-1] == latest_fy]
print(f"Latest seqs {len(latest_seqs)}")

pf_list = []
tf_list = []
pd_list = []
td_list = []
entry_list = []
tick_list = []
embeddings = []
with torch.no_grad():
    for seq in latest_seqs:
        xs_seq, ms_seq, te_seq, yn_seq, vm, sector_ids, raw = build_batch_one(seq)
        xs_t = {fam: torch.tensor(xs_seq[fam]) for fam in fams}
        ms_t = {fam: torch.tensor(ms_seq[fam]) for fam in fams}
        te_t = torch.tensor(te_seq)
        yn_t = torch.tensor(yn_seq)
        vm_t = torch.tensor(vm)
        c_seq, z_seq, out = model.forward_sequence(xs_t, ms_t, te_t, yn_t, vm_t)
        pos = len(seq["indices"]) - 1
        if pos < 0:
            continue
        pf = out["fwd_ret"][0, pos, 2].item()
        pd_ = out["fwd_dd"][0, pos].item()
        entry_logit = out["entry"][0, pos].item()
        entry_prob = 1 / (1 + np.exp(-entry_logit))
        orig_idx = seq["indices"][pos]
        tf = fwd["fwd_ret_6m"][orig_idx]
        td = fwd["fwd_dd_6m"][orig_idx]
        triple = fwd["triple_barrier"][orig_idx]
        pf_list.append(pf)
        tf_list.append(tf)
        pd_list.append(pd_)
        td_list.append(td)
        tick_list.append(seq["ticker"])
        emb = c_seq[0, pos, :].cpu().numpy()
        embeddings.append(emb)
        entry_list.append((seq["ticker"], pf, pd_, entry_prob, tf, td, triple))

pf_arr = np.array(pf_list)
tf_arr = np.array(tf_list)
pd_arr = np.array(pd_list)
td_arr = np.array(td_list)
print(
    f"PF mean {pf_arr.mean():.4f} std {pf_arr.std():.4f} min {pf_arr.min():.4f} max {pf_arr.max():.4f}"
)
print(f"TF mean {np.nanmean(tf_arr):.4f} std {np.nanstd(tf_arr):.4f}")
print(f"PD mean {pd_arr.mean():.4f} std {pd_arr.std():.4f}")
print(f"TD mean {np.nanmean(td_arr):.4f} std {np.nanstd(td_arr):.4f}")
from scipy.stats import spearmanr

ic = spearmanr(pf_arr, tf_arr, nan_policy="omit")[0]
ic_dd = spearmanr(pd_arr, td_arr, nan_policy="omit")[0]
print(f"IC f6 {ic:.4f} dd IC {ic_dd:.4f}")

# trades
trades = [e for e in entry_list if e[3] >= 0.7 and e[1] >= 0.05 and e[2] >= -0.25]
print(f"Adaptive trades entry>=0.7 fwd>=0.05 dd>=-0.25: {len(trades)}")
for t in trades[:20]:
    print(t)

scored = [
    (
        ticker,
        fwd_pred * entry_prob - dd_pred * 0.2,
        fwd_pred,
        dd_pred,
        entry_prob,
        true_fwd,
        true_dd,
    )
    for ticker, fwd_pred, dd_pred, entry_prob, true_fwd, true_dd, _ in entry_list
]
scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
print("Top 20 scored:")
for s in scored_sorted[:20]:
    print(s)

np.savez_compressed(
    DATA_DIR / "embedding_career_v6.npz",
    embedding=np.stack(embeddings),
    tickers=tick_list,
)
print("Saved embedding")

import pandas as pd

df = pd.DataFrame(
    [
        {
            "ticker": t[0],
            "pred_fwd": t[1],
            "pred_dd": t[2],
            "entry_prob": t[3],
            "true_fwd": t[4],
            "true_dd": t[5],
            "triple": t[6],
            "score": t[1] * t[3] - t[2] * 0.2,
        }
        for t in entry_list
    ]
)
df = df.sort_values("score", ascending=False)
df.to_csv(DATA_DIR / "trades_final_ranked_v6.csv", index=False)
print("Saved trades csv")
print(f"Absolute pred>=5% count {(pf_arr >= 0.05).sum()} total {len(pf_arr)}")
