import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "pipeline")
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


def build_model(deeper=False):
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
    if deeper:
        import torch.nn as nn

        model.entry_head = nn.Sequential(
            nn.Linear(args["dim"], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    model.load_state_dict(ckpt["model"], strict=False)
    return model


def build_batch(batch_list):
    B = len(batch_list)
    L = 10
    xs_seq = {fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    ms_seq = {fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
    te_seq = np.zeros((B, L, 8), dtype=np.float32)
    yn_seq = np.zeros((B, L, 1), dtype=np.float32)
    vm = np.zeros((B, L), dtype=bool)
    triple = np.full((B, L), -1, dtype=np.int32)
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
            triple[b, pos] = int(fwd["triple_barrier"][orig_idx])
        for pos, orig_idx in enumerate(seq["indices"]):
            if pos >= L:
                break
            for fam, cols in fams.items():
                xs_seq[fam][b, pos, :] = Z[orig_idx, cols]
                ms_seq[fam][b, pos, :] = mask[orig_idx, cols]
    return xs_seq, ms_seq, te_seq, yn_seq, vm, triple


def evaluate_last(model):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            batch_list = seqs[i : i + 32]
            xs, ms, te, yn, vm, _triple = build_batch(batch_list)
            xs_t = {fam: torch.tensor(xs[fam]) for fam in fams}
            ms_t = {fam: torch.tensor(ms[fam]) for fam in fams}
            te_t = torch.tensor(te)
            yn_t = torch.tensor(yn)
            vm_t = torch.tensor(vm)
            _c_seq, _z_seq, out = model.forward_sequence(xs_t, ms_t, te_t, yn_t, vm_t)
            entry = torch.sigmoid(out["entry"]).numpy()
            for b in range(len(batch_list)):
                if not vm[b].any():
                    continue
                last = np.where(vm[b])[0][-1]
                preds.append(entry[b, last])
    preds = np.array(preds)
    return preds.mean(), preds.std(), preds.min(), preds.max()


configs = [
    {
        "name": "bce_pw3.5_w4_gamma0",
        "pw": 3.5,
        "gamma": 0.0,
        "w": 4.0,
        "lr": 0.005,
        "epochs": 15,
        "deeper": False,
    },
    {
        "name": "bce_pw4.0_w4_gamma0",
        "pw": 4.0,
        "gamma": 0.0,
        "w": 4.0,
        "lr": 0.005,
        "epochs": 15,
        "deeper": False,
    },
    {
        "name": "bce_pw3.0_w5_gamma0_deep64",
        "pw": 3.0,
        "gamma": 0.0,
        "w": 5.0,
        "lr": 0.003,
        "epochs": 18,
        "deeper": True,
    },
    {
        "name": "bce_pw5.0_w5_gamma0",
        "pw": 5.0,
        "gamma": 0.0,
        "w": 5.0,
        "lr": 0.005,
        "epochs": 12,
        "deeper": False,
    },
]

for cfg in configs:
    print(f"\n=== {cfg['name']} ===")
    model = build_model(deeper=cfg["deeper"])
    for n, p in model.named_parameters():
        p.requires_grad = "entry_head" in n
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable}")
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"]
    )
    best_max = 0
    for ep in range(cfg["epochs"]):
        model.train()
        tot = 0
        nb = 0
        for i in range(0, len(seqs), 16):
            batch_list = seqs[i : i + 16]
            xs, ms, te, yn, vm, triple = build_batch(batch_list)
            xs_t = {fam: torch.tensor(xs[fam]) for fam in fams}
            ms_t = {fam: torch.tensor(ms[fam]) for fam in fams}
            te_t = torch.tensor(te)
            yn_t = torch.tensor(yn)
            vm_t = torch.tensor(vm)
            triple_t = torch.tensor(triple)
            opt.zero_grad()
            c_seq, z_seq, out = model.forward_sequence(xs_t, ms_t, te_t, yn_t, vm_t)
            logits = out["entry"]
            mask_t = (triple_t != -1) & vm_t
            if not mask_t.any():
                continue
            target = (triple_t[mask_t] == 1).float()
            logit = logits[mask_t]
            pw = torch.tensor([cfg["pw"]])
            bce = F.binary_cross_entropy_with_logits(
                logit, target, pos_weight=pw, reduction="none"
            )
            if cfg["gamma"] > 0:
                prob = torch.sigmoid(logit)
                pt = torch.where(target == 1, prob, 1 - prob)
                focal = (1 - pt).pow(cfg["gamma"])
                bce = (focal * bce).mean()
            else:
                bce = bce.mean()
            loss = bce * cfg["w"]
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        mean, std, mn, mx = evaluate_last(model)
        print(
            f"Ep {ep + 1}/{cfg['epochs']} loss {tot / max(1, nb):.4f} mean {mean:.3f} std {std:.3f} min {mn:.3f} max {mx:.3f} spread {mx - mn:.3f}"
        )
        if mx > best_max:
            best_max = mx
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "mean": mean, "max": mx},
                f"pipeline/data/mtnn_entry_tuned_{cfg['name']}.pt",
            )
    print(f"Best max for {cfg['name']}: {best_max}")
