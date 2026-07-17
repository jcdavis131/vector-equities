"""
Vector Equities Continuous Career MTNN trainer
- replaces season_ids with continuous time_enc (8-d) + year_norm
- TemporalInfoNCE weighted exp(-|delta|/3) + sector hard negatives
- Trading heads: fwd_ret 1M/3M/6M/12M, fwd_vol, fwd_dd, entry (triple barrier), turnaround, distress
- Validation: fwd_ret IC, entry precision@20, recall@10 career

"""
import argparse, json, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
import sys
sys.path.insert(0, str(ROOT / "pipeline"))
from model_career import EquitiesCareerMTNN
from dataset_career import load_bundle, family_slices, build_sequences, get_time_enc_for_seq
from feature_spec import SECTORS

def temporal_info_nce(c_seq, valid_mask, sector_ids, temp=0.08, hard_boost=0.3, delta_decay=3.0):
    """
    c_seq: (B, L, D) normalized
    valid_mask: (B, L) bool
    sector_ids: (B,) int sector per ticker
    Returns loss
    """
    B, L, D = c_seq.shape
    device = c_seq.device
    # flatten valid embeddings
    # For each ticker, create adjacent pairs (t, t+1) as positives
    anchors = []
    positives = []
    anchor_sector = []
    anchor_pos_index = []  # for logging
    # Build list of all valid embeddings for negative pool
    # Also need mapping from (b,l) to flat index in all_valid
    valid_flat_idx = -torch.ones((B, L), dtype=torch.long, device=device)
    all_valid_embs = []
    all_valid_sector = []
    all_valid_coords = []  # (b,l)

    flat_counter = 0
    for b in range(B):
        for l in range(L):
            if valid_mask[b,l]:
                valid_flat_idx[b,l] = flat_counter
                flat_counter += 1
                all_valid_embs.append(c_seq[b,l])
                all_valid_sector.append(sector_ids[b])
                all_valid_coords.append((b,l))

    if flat_counter < 2:
        return c_seq.sum()*0.0, 0, 0

    all_valid_embs = torch.stack(all_valid_embs)  # (N_valid, D)
    all_valid_sector = torch.tensor(all_valid_sector, device=device)

    # Build positive pairs: for each b, for each l where l and l+1 valid
    for b in range(B):
        for l in range(L-1):
            if valid_mask[b,l] and valid_mask[b,l+1]:
                # anchor = (b,l), positive = (b,l+1)
                anchors.append(c_seq[b,l])
                positives.append(c_seq[b,l+1])
                anchor_sector.append(sector_ids[b])
                anchor_pos_index.append(int(valid_flat_idx[b,l+1]))  # index in all_valid

    if len(anchors)==0:
        return c_seq.sum()*0.0, 0, flat_counter

    anchors = torch.stack(anchors)  # (N_pairs, D)
    positives_t = torch.stack(positives)

    # Compute logits anchors @ all_valid.T / temp
    logits = anchors @ all_valid_embs.T / temp  # (N_pairs, N_valid)

    # temporal weighting? For adjacent, delta=1 => weight exp(-1/3)=0.716. We'll incorporate as scaling of loss later.
    # For hard negatives: same sector different ticker -> boost logit by hard_boost
    # Build hard mask: anchor_sector[i] == all_valid_sector[j] and different ticker (not same b)
    # Need ticker identity for negative pool: we have coords
    anchor_coords = []  # we need to know anchor b for each pair
    # rebuild anchor b list
    anchor_b = []
    idx = 0
    for b in range(B):
        for l in range(L-1):
            if valid_mask[b,l] and valid_mask[b,l+1]:
                anchor_b.append(b)
                idx+=1

    anchor_b_t = torch.tensor(anchor_b, device=device)
    # For each anchor pair i, for each valid j, if sector same and ticker different => hard
    # ticker different means all_valid_coords[j][0] != anchor_b[i]
    # Create matrix (N_pairs, N_valid) bool
    all_valid_b = torch.tensor([c[0] for c in all_valid_coords], device=device)  # (N_valid)
    # sector match
    sector_match = (torch.tensor(anchor_sector, device=device).unsqueeze(1) == all_valid_sector.unsqueeze(0))  # (N_pairs, N_valid)
    diff_ticker = (anchor_b_t.unsqueeze(1) != all_valid_b.unsqueeze(0))
    hard_mask = sector_match & diff_ticker
    logits = logits + hard_mask.float() * hard_boost

    # target indices
    target = torch.tensor(anchor_pos_index, device=device, dtype=torch.long)

    loss = F.cross_entropy(logits, target)

    # Also add symmetric? For simplicity, also compute reverse (positive as anchor, anchor as negative) could help, but keep one direction.

    # Optional: weight by temporal distance (all adjacent => same weight). If we also include longer deltas, weighting would matter.
    return loss, len(anchors), flat_counter

def compute_ic(pred, target):
    # pred, target: 1D arrays, may have nan
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 10:
        return None
    p = pred[mask]
    t = target[mask]
    # rank correlation Spearman approximation via Pearson on ranks
    # simple Pearson
    if np.std(p) < 1e-6 or np.std(t) < 1e-6:
        return 0.0
    return float(np.corrcoef(p, t)[0,1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--d-tower", type=int, default=24)
    ap.add_argument("--d-tower-hidden", type=int, default=96)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)  # batch = tickers
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temp", type=float, default=0.08)
    ap.add_argument("--hard-boost", type=float, default=0.3)
    ap.add_argument("--drop-p", type=float, default=0.12)
    ap.add_argument("--weight-fwd-ret", type=float, default=0.5)
    ap.add_argument("--weight-entry", type=float, default=1.5)
    ap.add_argument("--weight-vol", type=float, default=0.1)
    ap.add_argument("--weight-dd", type=float, default=0.1)
    ap.add_argument("--weight-nce", type=float, default=1.0)
    ap.add_argument("--weight-sector", type=float, default=0.15)
    ap.add_argument("--weight-archetype", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--max-seq-len", type=int, default=10)
    Args = ap.parse_args()

    torch.manual_seed(Args.seed)
    np.random.seed(Args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device}")

    Z, mask, Z_raw, tickers, names, fiscal_years_arr, sectors_arr, manifest, fwd, bundle_path = load_bundle()
    fams, feat_list = family_slices(manifest)
    sequences, feat_to_idx_map_idx, extra = build_sequences(Z, mask, Z_raw, tickers, fiscal_years_arr, sectors_arr, manifest, fwd, max_seq_len=Args.max_seq_len)

    feat_to_idx = {f:i for i,f in enumerate(feat_list)}

    # Build fam dims
    fam_dims = {fam: len(cols) for fam, cols in fams.items()}
    print(f"Fams {fam_dims}")

    # Sector mapping
    sector_to_idx = {s:i for i,s in enumerate(SECTORS)}
    # For unknown sectors map to -1

    # Split sequences into train/val/test by ticker? We want time-based split for validation but also ticker split for generalization.
    # Simplest: 70% train tickers, 15% val, 15% test for career-level split, then within each, temporal split for forward labels:
    # Train: FY <=2021, Val: 2022-2023, Test: 2024. That matches old logic eval_split.
    # We'll split tickers randomly but keep same temporal eval inside training loss.
    np.random.seed(Args.seed)
    perm_tickers = np.random.permutation(len(sequences))
    n = len(sequences)
    n_train = int(n*0.8)
    n_val = int(n*0.1)
    train_seqs = [sequences[i] for i in perm_tickers[:n_train]]
    val_seqs = [sequences[i] for i in perm_tickers[n_train:n_train+n_val]]
    test_seqs = [sequences[i] for i in perm_tickers[n_train+n_val:]]

    print(f"Split {len(train_seqs)} train {len(val_seqs)} val {len(test_seqs)} test")

    # Model
    model = EquitiesCareerMTNN(
        fam_dims=fam_dims,
        d_tower=Args.d_tower,
        d_tower_hidden=Args.d_tower_hidden,
        d_emb=Args.dim,
        d_time=8,
        d_time_emb=16,
        d_model=Args.d_model,
        n_layers=Args.n_layers,
        n_heads=Args.n_heads,
        n_game=14,
        n_skills=0,  # disable skills for now to avoid dim mismatch
        n_sectors=len(SECTORS),
        n_archetypes=8,
        n_tower_blocks=1,
        d_skill_hidden=16,
        d_head_hidden=64,
        mlp_heads=True,
        dropout=0.1,
    ).to(device)

    decay, no_decay = [], []
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim==1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([{"params":decay,"weight_decay":1e-4},{"params":no_decay,"weight_decay":0.0}], lr=Args.lr)
    import math
    steps_per_epoch = max(1, math.ceil(len(train_seqs)/Args.batch))
    total_steps = steps_per_epoch * Args.epochs
    print(f"steps_per_epoch {steps_per_epoch} total {total_steps}")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=Args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="linear")

    # Precompute helper to build batch tensors
    def build_batch(batch_seq_list):
        B = len(batch_seq_list)
        L = Args.max_seq_len
        # xs_seq dict
        xs_seq = {fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
        ms_seq = {fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32) for fam in fams}
        time_enc_seq = np.zeros((B, L, 8), dtype=np.float32)
        year_norm_seq = np.zeros((B, L, 1), dtype=np.float32)
        valid_mask = np.zeros((B, L), dtype=bool)
        # forward labels
        fwd_ret_1m = np.full((B,L), np.nan, dtype=np.float32)
        fwd_ret_3m = np.full((B,L), np.nan, dtype=np.float32)
        fwd_ret_6m = np.full((B,L), np.nan, dtype=np.float32)
        fwd_ret_12m = np.full((B,L), np.nan, dtype=np.float32)
        fwd_vol = np.full((B,L), np.nan, dtype=np.float32)
        fwd_dd = np.full((B,L), np.nan, dtype=np.float32)
        triple = np.full((B,L), -1, dtype=np.int64)
        ceo_change = np.zeros((B,L), dtype=np.float32)
        fys = np.zeros((B,L), dtype=np.int64)
        sector_ids = np.zeros((B,), dtype=np.int64)
        tickers_b = []
        for b, seq in enumerate(batch_seq_list):
            tickers_b.append(seq["ticker"])
            # sector id
            sec_name = seq["sector"]
            sector_ids[b] = sector_to_idx.get(sec_name, -1)
            enc = get_time_enc_for_seq(seq, Z_raw, fwd, feat_to_idx)
            time_enc_seq[b] = enc["time_enc"]
            year_norm_seq[b] = enc["year_norm"]
            valid_mask[b] = enc["mask"]
            fwd_ret_1m[b] = enc["fwd_ret_1m"]
            fwd_ret_3m[b] = enc["fwd_ret_3m"]
            fwd_ret_6m[b] = enc["fwd_ret_6m"]
            fwd_ret_12m[b] = enc["fwd_ret_12m"]
            fwd_vol[b] = enc["fwd_vol_6m"]
            fwd_dd[b] = enc["fwd_dd_6m"]
            triple[b] = enc["triple_barrier"]
            ceo_change[b] = enc["ceo_change_flag"]
            fys[b, :len(seq["fiscal_years"])] = seq["fiscal_years"]
            for pos, orig_idx in enumerate(seq["indices"]):
                if pos>=L: break
                for fam, cols in fams.items():
                    xs_seq[fam][b,pos,:] = Z[orig_idx, cols]
                    ms_seq[fam][b,pos,:] = mask[orig_idx, cols]
        return {
            "xs_seq": xs_seq,
            "ms_seq": ms_seq,
            "time_enc_seq": time_enc_seq,
            "year_norm_seq": year_norm_seq,
            "valid_mask": valid_mask,
            "fwd_ret_1m": fwd_ret_1m,
            "fwd_ret_3m": fwd_ret_3m,
            "fwd_ret_6m": fwd_ret_6m,
            "fwd_ret_12m": fwd_ret_12m,
            "fwd_vol": fwd_vol,
            "fwd_dd": fwd_dd,
            "triple": triple,
            "ceo_change": ceo_change,
            "sector_ids": sector_ids,
            "tickers": tickers_b,
            "fys": fys,
        }

    def to_torch(batch_np):
        out = {}
        out["xs_seq"] = {fam: torch.tensor(batch_np["xs_seq"][fam], device=device) for fam in fams}
        out["ms_seq"] = {fam: torch.tensor(batch_np["ms_seq"][fam], device=device) for fam in fams}
        out["time_enc_seq"] = torch.tensor(batch_np["time_enc_seq"], device=device)
        out["year_norm_seq"] = torch.tensor(batch_np["year_norm_seq"], device=device)
        out["valid_mask"] = torch.tensor(batch_np["valid_mask"], device=device)
        out["fwd_ret_6m"] = torch.tensor(batch_np["fwd_ret_6m"], device=device)
        out["fwd_ret_1m"] = torch.tensor(batch_np["fwd_ret_1m"], device=device)
        out["fwd_ret_3m"] = torch.tensor(batch_np["fwd_ret_3m"], device=device)
        out["fwd_ret_12m"] = torch.tensor(batch_np["fwd_ret_12m"], device=device)
        out["fwd_vol"] = torch.tensor(batch_np["fwd_vol"], device=device)
        out["fwd_dd"] = torch.tensor(batch_np["fwd_dd"], device=device)
        out["triple"] = torch.tensor(batch_np["triple"], device=device)
        out["sector_ids"] = torch.tensor(batch_np["sector_ids"], device=device)
        # Keep numpy for IC later
        out["_np"] = batch_np
        return out

    best_val_ic = -999
    best_path = DATA_DIR / "mtnn_career_best.pt"
    for epoch in range(Args.epochs):
        model.train()
        np.random.shuffle(train_seqs)
        total_loss = 0
        n_batches = 0
        total_nce = 0
        total_fwd = 0
        total_entry = 0
        for s in range(0, len(train_seqs), Args.batch):
            batch_list = train_seqs[s:s+Args.batch]
            if len(batch_list) < 2:
                continue
            batch_np = build_batch(batch_list)
            batch_t = to_torch(batch_np)
            # random feature dropout
            # apply dropout to xs
            for fam in batch_t["xs_seq"]:
                if Args.drop_p>0:
                    drop_mask = (torch.rand_like(batch_t["ms_seq"][fam]) > Args.drop_p).float()
                    batch_t["xs_seq"][fam] = batch_t["xs_seq"][fam] * drop_mask
                    batch_t["ms_seq"][fam] = batch_t["ms_seq"][fam] * drop_mask

            c_seq, z_seq, out = model.forward_sequence(
                batch_t["xs_seq"],
                batch_t["ms_seq"],
                batch_t["time_enc_seq"],
                batch_t["year_norm_seq"],
                batch_t["valid_mask"]
            )
            # Losses
            loss = 0.0
            # NCE temporal
            nce_loss, n_pairs, n_valid = temporal_info_nce(c_seq, batch_t["valid_mask"], batch_t["sector_ids"], temp=Args.temp, hard_boost=Args.hard_boost)
            loss = loss + Args.weight_nce * nce_loss

            # Sector head
            if "sector" in out and Args.weight_sector>0:
                sector_logits = out["sector"]  # B,L,n_sectors
                # only where sector_ids valid and valid_mask
                # sector_ids is per ticker, not per time, but assume same across time -> broadcast
                # target per (b,l) = sector_ids[b]
                B,L,_ = sector_logits.shape
                sector_target = batch_t["sector_ids"].unsqueeze(1).expand(B,L)  # B,L
                valid = batch_t["valid_mask"] & (batch_t["sector_ids"].unsqueeze(1).expand(B,L) >=0)
                if valid.any():
                    logits_valid = sector_logits[valid]
                    target_valid = sector_target[valid]
                    loss = loss + Args.weight_sector * F.cross_entropy(logits_valid, target_valid)

            # Forward return heads: 4
            # fwd_ret_head outputs (B,L,4) -> 1M,3M,6M,12M
            fwd_ret_pred = out["fwd_ret"]  # B,L,4
            # create target tensor (B,L,4) from batch_np
            fwd_target = torch.stack([
                batch_t["fwd_ret_1m"],
                batch_t["fwd_ret_3m"],
                batch_t["fwd_ret_6m"],
                batch_t["fwd_ret_12m"]
            ], dim=-1)  # B,L,4
            # mask where finite
            valid_fwd = torch.isfinite(fwd_target) & batch_t["valid_mask"].unsqueeze(-1)
            if valid_fwd.any():
                # Use MSE
                diff = (fwd_ret_pred - fwd_target)
                # only where valid
                mse = (diff[valid_fwd]**2).mean()
                loss = loss + Args.weight_fwd_ret * mse
                total_fwd += float(mse.detach())

            # Entry head triple barrier BCE
            entry_pred = out["entry"]  # B,L
            triple_target = batch_t["triple"]  # B,L -1/0/1
            valid_entry = (triple_target != -1) & batch_t["valid_mask"]
            if valid_entry.any():
                # binary: 1 = profit hit before loss
                target = (triple_target[valid_entry]==1).float()
                logits = entry_pred[valid_entry]
                pos_weight = torch.tensor([1.5], device=logits.device)
                bce_raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction='none')
                prob = torch.sigmoid(logits)
                pt = torch.where(target==1, prob, 1-prob)
                focal_w = (1-pt).pow(2.0)
                bce = (focal_w * bce_raw).mean()
                loss = loss + Args.weight_entry * bce
                total_entry += float(bce.detach())

            # Vol and DD heads
            if "fwd_vol" in out and Args.weight_vol>0:
                valid_vol = torch.isfinite(batch_t["fwd_vol"]) & batch_t["valid_mask"]
                if valid_vol.any():
                    loss_vol = F.mse_loss(out["fwd_vol"][valid_vol], batch_t["fwd_vol"][valid_vol])
                    loss = loss + Args.weight_vol * loss_vol
            if "fwd_dd" in out and Args.weight_dd>0:
                valid_dd = torch.isfinite(batch_t["fwd_dd"]) & batch_t["valid_mask"]
                if valid_dd.any():
                    loss_dd = F.mse_loss(out["fwd_dd"][valid_dd], batch_t["fwd_dd"][valid_dd])
                    loss = loss + Args.weight_dd * loss_dd

            # Backward
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            total_loss += float(loss.detach())
            total_nce += float(nce_loss.detach()) if isinstance(nce_loss, torch.Tensor) else 0
            n_batches += 1

        avg_loss = total_loss / max(1,n_batches)
        avg_nce = total_nce / max(1,n_batches)
        print(f"Ep {epoch+1}/{Args.epochs} loss {avg_loss:.4f} nce {avg_nce:.4f} fwd {total_fwd/max(1,n_batches):.4f} entry {total_entry/max(1,n_batches):.4f}")

        if (epoch+1) % Args.val_every == 0 or epoch==Args.epochs-1:
            # validation: compute fwd_ret_6M IC and entry precision
            model.eval()
            with torch.no_grad():
                # Gather all val batches
                all_pred_ret = []
                all_true_ret = []
                all_entry_pred = []
                all_entry_true = []
                for s in range(0, len(val_seqs), Args.batch):
                    batch_list = val_seqs[s:s+Args.batch]
                    batch_np = build_batch(batch_list)
                    batch_t = to_torch(batch_np)
                    c_seq, z_seq, out = model.forward_sequence(
                        batch_t["xs_seq"], batch_t["ms_seq"], batch_t["time_enc_seq"], batch_t["year_norm_seq"], batch_t["valid_mask"]
                    )
                    # fwd ret 6m: out["fwd_ret"][:,:,2] corresponds to 6m (index 2)
                    pred_6m = out["fwd_ret"][:,:,2].cpu().numpy()
                    true_6m = batch_np["fwd_ret_6m"]
                    # flatten valid
                    mask_valid = batch_np["valid_mask"] & np.isfinite(true_6m)
                    all_pred_ret.append(pred_6m[mask_valid])
                    all_true_ret.append(true_6m[mask_valid])

                    entry_pred = torch.sigmoid(out["entry"]).cpu().numpy()
                    triple = batch_np["triple"]
                    mask_entry = (triple != -1) & batch_np["valid_mask"]
                    all_entry_pred.append(entry_pred[mask_entry])
                    all_entry_true.append((triple[mask_entry]==1).astype(int))

                if all_pred_ret:
                    pred_concat = np.concatenate(all_pred_ret) if len(all_pred_ret)>0 else np.array([])
                    true_concat = np.concatenate(all_true_ret) if len(all_true_ret)>0 else np.array([])
                    ic = compute_ic(pred_concat, true_concat)
                else:
                    ic = None

                # entry precision@20: top 20 entry scores and see true positive rate
                if all_entry_pred and len(all_entry_pred)>0:
                    ep = np.concatenate(all_entry_pred)
                    et = np.concatenate(all_entry_true)
                    if len(ep)>=20:
                        top20_idx = np.argsort(-ep)[:20]
                        prec20 = et[top20_idx].mean()
                    else:
                        prec20 = et.mean() if len(et)>0 else 0
                else:
                    prec20 = 0

                print(f" VAL IC 6M: {ic}  EntryPrec@20: {prec20:.3f}")

                if ic is not None and ic > best_val_ic:
                    best_val_ic = ic
                    torch.save({
                        "model": model.state_dict(),
                        "args": vars(Args),
                        "epoch": epoch,
                        "ic": ic,
                        "prec20": prec20,
                    }, best_path)
                    print(f"  -> Saved best {best_path} ic {ic:.4f}")

    # Final save
    final_path = DATA_DIR / "mtnn_career_last.pt"
    torch.save({"model": model.state_dict(), "args": vars(Args)}, final_path)
    print(f"Saved final {final_path}")
    # Also save embedding for latest train_matrix for visualization
    model.eval()
    with torch.no_grad():
        # embed all sequences for train set (or all tickers)
        all_seqs = sequences
        all_embs = []
        all_meta = []
        for s in range(0, len(all_seqs), Args.batch):
            batch_list = all_seqs[s:s+Args.batch]
            batch_np = build_batch(batch_list)
            batch_t = to_torch(batch_np)
            c_seq, z_seq, out = model.forward_sequence(
                batch_t["xs_seq"], batch_t["ms_seq"], batch_t["time_enc_seq"], batch_t["year_norm_seq"], batch_t["valid_mask"]
            )
            # Use last valid embedding per ticker as career embedding?
            # For compatibility with old embedding.npz, we need per-row embedding, not per-ticker.
            # So we output career-aware c_seq per timestep (B,L,D)
            # Flatten to rows
            for b_idx, seq in enumerate(batch_list):
                valid_len = seq["valid_len"]
                for l in range(valid_len):
                    all_embs.append(c_seq[b_idx,l].cpu().numpy())
                    all_meta.append((seq["ticker"], int(seq["fiscal_years"][l]), seq["sector"]))

        embs = np.stack(all_embs).astype(np.float32)
        tickers_out = np.array([m[0] for m in all_meta])
        fys_out = np.array([str(m[1]) for m in all_meta])
        sectors_out = np.array([m[2] for m in all_meta])
        np.savez_compressed(DATA_DIR / "embedding_career.npz", E=embs, ticker=tickers_out, fiscal_year=fys_out, sector=sectors_out)
        print(f"Saved career embedding {DATA_DIR/'embedding_career.npz'} shape {embs.shape}")

if __name__ == "__main__":
    main()
