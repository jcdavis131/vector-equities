import sys

sys.path.insert(0, "pipeline")
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dataset_career import (
    build_sequences,
    family_slices,
    get_time_enc_for_seq,
    load_bundle,
)
from model_career import EquitiesCareerMTNN

DATA_DIR = Path("pipeline/data")
Z, mask, Z_raw, tickers_b, names, fy_arr, sectors_arr, manifest, fwd, _ = load_bundle()
fams, feat_list = family_slices(manifest)
fam_dims = {fam: len(c) for fam, c in fams.items()}
seqs, _, _ = build_sequences(
    Z, mask, Z_raw, tickers_b, fy_arr, sectors_arr, manifest, fwd
)

ckpt = torch.load(
    DATA_DIR / "mtnn_career_best.pt", map_location="cpu", weights_only=False
)
args = ckpt["args"]


def build_model():
    m = EquitiesCareerMTNN(
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
    m.load_state_dict(ckpt["model"], strict=False)
    return m


def build_batch(batch_list):
    B = len(batch_list)
    L = 10
    xs_seq = {fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    ms_seq = {fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    te_seq = np.zeros((B, L, 8), dtype=np.float32)
    yn_seq = np.zeros((B, L, 1), dtype=np.float32)
    vm = np.zeros((B, L), dtype=bool)
    fwd6 = np.full((B, L), np.nan, dtype=np.float32)
    fdd = np.full((B, L), np.nan, dtype=np.float32)
    fvol = np.full((B, L), np.nan, dtype=np.float32)
    for b, seq in enumerate(batch_list):
        enc = get_time_enc_for_seq(
            seq, Z_raw, fwd, {f: i for i, f in enumerate(feat_list)}
        )
        te_seq[b] = enc["time_enc"]
        yn_seq[b] = enc["year_norm"]
        vm[b] = enc["mask"]
        for pos, orig_idx in enumerate(seq["indices"]):
            if pos >= L:
                break
            fwd6[b, pos] = fwd["fwd_ret_6m"][orig_idx]
            fdd[b, pos] = fwd["fwd_dd_6m"][orig_idx]
            fvol[b, pos] = fwd["fwd_vol_6m"][orig_idx]
        for pos, orig_idx in enumerate(seq["indices"]):
            if pos >= L:
                break
            for fam, cols in fams.items():
                xs_seq[fam][b, pos, :] = Z[orig_idx, cols]
                ms_seq[fam][b, pos, :] = mask[orig_idx, cols]
    return xs_seq, ms_seq, te_seq, yn_seq, vm, fwd6, fdd, fvol


model = build_model()
# freeze all except fwd_ret, fwd_dd, fwd_vol heads
for n, p in model.named_parameters():
    p.requires_grad = any(
        k in n for k in ["fwd_ret_head", "fwd_dd_head", "fwd_vol_head"]
    )
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable fwd heads {trainable}")

opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.003, weight_decay=1e-4
)


# For eval
def eval_stats():
    model.eval()
    pf = []
    tf = []
    pd = []
    td = []
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            bl = seqs[i : i + 32]
            xs, ms, te, yn, vm, f6, fd, _fv = build_batch(bl)
            xs_t = {fam: torch.tensor(xs[fam]) for fam in fams}
            ms_t = {fam: torch.tensor(ms[fam]) for fam in fams}
            te_t = torch.tensor(te)
            yn_t = torch.tensor(yn)
            _c_seq, _z_seq, out = model.forward_sequence(
                xs_t, ms_t, te_t, yn_t, torch.tensor(vm)
            )
            pred_f6 = out["fwd_ret"][:, :, 2].numpy()
            pred_dd = out["fwd_dd"].numpy()
            for b in range(len(bl)):
                for seq_pos in range(bl[b]["valid_len"]):
                    if np.isnan(f6[b, seq_pos]) or np.isnan(fd[b, seq_pos]):
                        continue
                    pf.append(pred_f6[b, seq_pos])
                    tf.append(f6[b, seq_pos])
                    pd.append(pred_dd[b, seq_pos])
                    td.append(fd[b, seq_pos])
    pf = np.array(pf)
    tf = np.array(tf)
    pd = np.array(pd)
    td = np.array(td)
    # IC
    from scipy.stats import spearmanr

    ic_f6 = spearmanr(pf, tf)[0]
    ic_dd = spearmanr(pd, td)[0]
    mean_f6 = pf.mean()
    std_f6 = pf.std()
    mean_dd = pd.mean()
    std_dd = pd.std()
    return mean_f6, std_f6, ic_f6, mean_dd, std_dd, ic_dd, pf, tf, pd, td


configs = [
    {
        "name": "fwd_w3_dd_w1_rank0.5_var0.1",
        "w_f6": 3.0,
        "w_dd": 1.0,
        "w_vol": 0.2,
        "rank_w": 0.5,
        "var_w": 0.1,
        "lr": 0.003,
        "epochs": 12,
    },
    {
        "name": "fwd_w5_dd_w1.5_rank1_var0.2",
        "w_f6": 5.0,
        "w_dd": 1.5,
        "w_vol": 0.3,
        "rank_w": 1.0,
        "var_w": 0.2,
        "lr": 0.002,
        "epochs": 12,
    },
]

for cfg in configs:
    print(f"\n=== {cfg['name']} ===")
    # reload fresh model each cfg
    model = build_model()
    for n, p in model.named_parameters():
        p.requires_grad = any(
            k in n for k in ["fwd_ret_head", "fwd_dd_head", "fwd_vol_head"]
        )
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"]
    )
    best_ic = 0
    for ep in range(cfg["epochs"]):
        model.train()
        tot_loss = 0
        nb = 0
        for i in range(0, len(seqs), 32):
            bl = seqs[i : i + 32]
            xs, ms, te, yn, vm, f6, fd, fv = build_batch(bl)
            xs_t = {fam: torch.tensor(xs[fam]) for fam in fams}
            ms_t = {fam: torch.tensor(ms[fam]) for fam in fams}
            te_t = torch.tensor(te)
            yn_t = torch.tensor(yn)
            vm_t = torch.tensor(vm)
            t_f6 = torch.tensor(f6)
            t_dd = torch.tensor(fd)
            t_vol = torch.tensor(fv)
            opt.zero_grad()
            c_seq, z_seq, out = model.forward_sequence(xs_t, ms_t, te_t, yn_t, vm_t)
            pred_f6 = out["fwd_ret"][:, :, 2]
            pred_dd = out["fwd_dd"]
            pred_vol = out["fwd_vol"]
            mask_valid = torch.tensor(~np.isnan(f6))
            if mask_valid.sum() == 0:
                continue
            # MSE
            mse_f6 = F.mse_loss(pred_f6[mask_valid], t_f6[mask_valid])
            mse_dd = F.mse_loss(pred_dd[mask_valid], t_dd[mask_valid])
            mask_vol = torch.tensor(~np.isnan(fv))
            mse_vol = (
                F.mse_loss(pred_vol[mask_vol], t_vol[mask_vol])
                if mask_vol.sum() > 0
                else torch.tensor(0.0)
            )
            loss = cfg["w_f6"] * mse_f6 + cfg["w_dd"] * mse_dd + cfg["w_vol"] * mse_vol

            # ranking loss: pairwise margin for f6
            if cfg["rank_w"] > 0 and mask_valid.sum() > 5:
                # flatten valid
                pf_v = pred_f6[mask_valid]
                tf_v = t_f6[mask_valid]
                # sample 128 random pairs to keep cheap
                n_pairs = min(256, len(pf_v) * 2)
                idx1 = torch.randint(0, len(pf_v), (n_pairs,))
                idx2 = torch.randint(0, len(pf_v), (n_pairs,))
                # true diff
                true_diff = tf_v[idx1] - tf_v[idx2]
                # we only care where |true_diff| > 0.05 (5% diff)
                mask_pair = torch.abs(true_diff) > 0.05
                if mask_pair.sum() > 0:
                    # desired order: if true_diff>0 then pred1>pred2
                    # use MarginRankingLoss with target sign
                    target = torch.sign(true_diff[mask_pair])
                    # pred diff
                    pred_diff = pf_v[idx1[mask_pair]] - pf_v[idx2[mask_pair]]
                    # ranking loss: max(0, margin - target*pred_diff)
                    margin = 0.02
                    rank_loss = torch.clamp(margin - target * pred_diff, min=0).mean()
                    loss = loss + cfg["rank_w"] * rank_loss

            # variance matching: encourage std close to true std
            if cfg["var_w"] > 0:
                std_pred_f6 = torch.std(pred_f6[mask_valid])
                std_true_f6 = torch.std(t_f6[mask_valid])
                var_loss_f6 = (std_pred_f6 - std_true_f6).pow(2)
                std_pred_dd = torch.std(pred_dd[mask_valid])
                std_true_dd = torch.std(t_dd[mask_valid])
                var_loss_dd = (std_pred_dd - std_true_dd).pow(2)
                loss = loss + cfg["var_w"] * (var_loss_f6 + var_loss_dd)

            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        mean_f6, std_f6, ic_f6, mean_dd, std_dd, ic_dd, _, _, _, _ = eval_stats()
        print(
            f"Ep {ep + 1} loss {tot_loss / max(1, nb):.4f} f6 mean {mean_f6:.3f} std {std_f6:.3f} IC {ic_f6:.3f} | dd mean {mean_dd:.3f} std {std_dd:.3f} IC {ic_dd:.3f}"
        )
        if ic_f6 > best_ic:
            best_ic = ic_f6
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "ic_f6": ic_f6,
                    "stats": (mean_f6, std_f6, mean_dd, std_dd),
                },
                f"pipeline/data/mtnn_fwd_tuned_{cfg['name']}.pt",
            )
    print(f"best ic {best_ic} for {cfg['name']}")
