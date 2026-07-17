import argparse, json, math, sys
sys.path.insert(0,"pipeline")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dataset_career import load_bundle, family_slices, build_sequences, get_time_enc_for_seq
from model_career import EquitiesCareerMTNN

def temporal_info_nce(c_seq, valid_mask, sector_ids, temp=0.08):
    B,L,D = c_seq.shape
    device=c_seq.device
    valid = valid_mask.reshape(-1)
    if valid.sum()<2:
        return torch.tensor(0.0, device=device), 0, 0
    c_valid = c_seq.reshape(-1,D)[valid]
    n_anchors = min(64, c_valid.size(0))
    idx = torch.randperm(c_valid.size(0), device=device)[:n_anchors]
    anchors = c_valid[idx]
    logits = anchors @ c_valid.T / temp
    loss = F.cross_entropy(logits, idx)
    return loss, n_anchors, int(valid.sum())

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--d-tower", type=int, default=24)
ap.add_argument("--d-tower-hidden", type=int, default=64)
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--max-seq-len", type=int, default=10)
ap.add_argument("--lr", type=float, default=0.002)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--temp", type=float, default=0.08)
ap.add_argument("--weight-fwd-ret", type=float, default=5.0)
ap.add_argument("--weight-entry", type=float, default=2.0)
ap.add_argument("--weight-vol", type=float, default=0.3)
ap.add_argument("--weight-dd", type=float, default=1.5)
ap.add_argument("--weight-nce", type=float, default=1.0)
ap.add_argument("--weight-sector", type=float, default=0.15)
ap.add_argument("--weight-rank", type=float, default=1.0)
ap.add_argument("--weight-var", type=float, default=0.2)
ap.add_argument("--drop-p", type=float, default=0.1)
ap.add_argument("--val-every", type=int, default=1)
ap.add_argument("--out", type=str, default="pipeline/data/mtnn_career_v6_best.pt")
Args = ap.parse_args()
print(f"Args {vars(Args)}")
torch.manual_seed(Args.seed)
np.random.seed(Args.seed)
DATA_DIR = Path("pipeline/data")
Z, mask, Z_raw, tickers, names, fiscal_years, sectors, manifest, fwd, npz_path = load_bundle()
print(f"Loaded {npz_path} Z {Z.shape}")
fams, feat_list = family_slices(manifest)
fam_dims = {fam: len(cols) for fam,cols in fams.items()}
print(f"Families {len(fams)}")
seqs, feat_to_idx, extra_idx = build_sequences(Z, mask, Z_raw, tickers, fiscal_years, sectors, manifest, fwd, max_seq_len=Args.max_seq_len)
print(f"Built {len(seqs)} seqs")
np.random.seed(Args.seed)
uniq_tickers = list(set([s["ticker"] for s in seqs]))
np.random.shuffle(uniq_tickers)
n = len(uniq_tickers)
n_train=int(n*0.7); n_val=int(n*0.15)
train_tickers=set(uniq_tickers[:n_train])
val_tickers=set(uniq_tickers[n_train:n_train+n_val])
train_seqs=[s for s in seqs if s["ticker"] in train_tickers]
val_seqs=[s for s in seqs if s["ticker"] in val_tickers]
print(f"Split train {len(train_seqs)} val {len(val_seqs)}")
device="cpu"
model = EquitiesCareerMTNN(
    fam_dims=fam_dims,
    d_tower=Args.d_tower,
    d_tower_hidden=Args.d_tower_hidden,
    d_emb=Args.dim,
    d_time=8, d_time_emb=16,
    d_model=Args.d_model,
    n_layers=Args.n_layers,
    n_heads=Args.n_heads,
    n_game=14, n_skills=0, n_sectors=11, n_archetypes=8, mlp_heads=True
).to(device)
decay=[]; no_decay=[]
for n,p in model.named_parameters():
    if "bias" in n or "norm" in n:
        no_decay.append(p)
    else:
        decay.append(p)
opt = torch.optim.AdamW([{"params":decay,"weight_decay":1e-4},{"params":no_decay,"weight_decay":0.0}], lr=Args.lr)
def build_batch(batch_list):
    B=len(batch_list); L=Args.max_seq_len
    xs_seq={fam: np.zeros((B,L,fam_dims[fam]), dtype=np.float32) for fam in fams}
    ms_seq={fam: np.zeros((B,L,fam_dims[fam]), dtype=np.float32) for fam in fams}
    te_seq=np.zeros((B,L,8), dtype=np.float32)
    yn_seq=np.zeros((B,L,1), dtype=np.float32)
    vm=np.zeros((B,L), dtype=bool)
    sector_ids=np.zeros((B,L), dtype=np.int64)
    fwd_6m=np.full((B,L), np.nan, dtype=np.float32)
    fwd_vol=np.full((B,L), np.nan, dtype=np.float32)
    fwd_dd=np.full((B,L), np.nan, dtype=np.float32)
    triple=np.full((B,L), -1, dtype=np.int64)
    for b,seq in enumerate(batch_list):
        enc=get_time_enc_for_seq(seq, Z_raw, fwd, feat_to_idx)
        te_seq[b]=enc["time_enc"]; yn_seq[b]=enc["year_norm"]; vm[b]=enc["mask"]
        sector_ids[b,:len(seq["indices"])] = seq.get("sector_id",0)
        for pos,orig_idx in enumerate(seq["indices"]):
            if pos>=L: break
            fwd_6m[b,pos]=fwd["fwd_ret_6m"][orig_idx]
            fwd_vol[b,pos]=fwd["fwd_vol_6m"][orig_idx]
            fwd_dd[b,pos]=fwd["fwd_dd_6m"][orig_idx]
            tb=fwd["triple_barrier"][orig_idx]
            if not np.isnan(tb):
                triple[b,pos]=int(tb)
        for pos,orig_idx in enumerate(seq["indices"]):
            if pos>=L: break
            for fam,cols in fams.items():
                xs_seq[fam][b,pos,:]=Z[orig_idx, cols]
                ms_seq[fam][b,pos,:]=mask[orig_idx, cols]
    return xs_seq, ms_seq, te_seq, yn_seq, vm, sector_ids, fwd_6m, fwd_vol, fwd_dd, triple
best_ic=-1; best_state=None
for epoch in range(Args.epochs):
    model.train()
    total_loss=0; total_nce=0; total_fwd=0; total_entry=0; nb=0
    for s in range(0, len(train_seqs), Args.batch):
        batch_list=train_seqs[s:s+Args.batch]
        xs_seq, ms_seq, te_seq, yn_seq, vm, sector_ids, f6,fvol,fdd,triple = build_batch(batch_list)
        xs_t={fam: torch.tensor(xs_seq[fam], device=device) for fam in fams}
        ms_t={fam: torch.tensor(ms_seq[fam], device=device) for fam in fams}
        te_t=torch.tensor(te_seq, device=device)
        yn_t=torch.tensor(yn_seq, device=device)
        vm_t=torch.tensor(vm, device=device)
        sector_t=torch.tensor(sector_ids, device=device)
        opt.zero_grad()
        c_seq, z_seq, out = model.forward_sequence(xs_t, ms_t, te_t, yn_t, vm_t)
        loss=torch.tensor(0.0, device=device)
        nce_loss, _, _ = temporal_info_nce(c_seq, vm_t, sector_t, temp=Args.temp)
        loss = loss + Args.weight_nce * nce_loss
        if "sector" in out and Args.weight_sector>0 and vm_t.sum()>0:
            logits = out["sector"][vm_t]
            target = sector_t[vm_t] % 11
            if logits.numel()>0:
                loss = loss + Args.weight_sector * F.cross_entropy(logits, target)
        t_f6=torch.tensor(f6, device=device)
        mask_fwd = ~torch.isnan(t_f6)
        if mask_fwd.sum()>0:
            mse = F.mse_loss(out["fwd_ret"][:,:,2][mask_fwd], t_f6[mask_fwd])
            loss = loss + Args.weight_fwd_ret * mse
            total_fwd+=float(mse.detach())
            if Args.weight_rank>0:
                pf = out["fwd_ret"][:,:,2][mask_fwd]
                tf = t_f6[mask_fwd]
                if pf.numel()>5:
                    n_pairs_rank = min(256, pf.numel()*2)
                    idx1 = torch.randint(0, pf.numel(), (n_pairs_rank,), device=device)
                    idx2 = torch.randint(0, pf.numel(), (n_pairs_rank,), device=device)
                    true_diff = tf[idx1]-tf[idx2]
                    mask_pair = torch.abs(true_diff)>0.05
                    if mask_pair.sum()>0:
                        target_sign = torch.sign(true_diff[mask_pair])
                        pred_diff = pf[idx1[mask_pair]]-pf[idx2[mask_pair]]
                        margin=0.02
                        rank_loss = torch.clamp(margin - target_sign*pred_diff, min=0).mean()
                        loss = loss + Args.weight_rank * rank_loss
            if Args.weight_var>0:
                pf_valid = out["fwd_ret"][:,:,2][mask_fwd]
                tf_valid = t_f6[mask_fwd]
                if pf_valid.numel()>2:
                    std_pred = torch.std(pf_valid)
                    std_true = torch.std(tf_valid)
                    var_loss = (std_pred - std_true).pow(2)
                    loss = loss + Args.weight_var * var_loss
        if "entry" in out and Args.weight_entry>0:
            logits = out["entry"][vm_t]
            tgt = torch.tensor(triple, device=device)[vm_t]
            valid_e = tgt!=-1
            if valid_e.sum()>0:
                logits_v = logits[valid_e]
                tgt_v = (tgt[valid_e]==1).float()
                pos_weight = torch.tensor([3.5], device=device)
                bce = F.binary_cross_entropy_with_logits(logits_v, tgt_v, pos_weight=pos_weight)
                loss = loss + Args.weight_entry * bce
                total_entry+=float(bce.detach())
        if "fwd_vol" in out and Args.weight_vol>0:
            tvol=torch.tensor(fvol, device=device)
            mask_vol = ~torch.isnan(tvol)
            if mask_vol.sum()>0:
                loss_vol = F.mse_loss(out["fwd_vol"][mask_vol], tvol[mask_vol])
                loss = loss + Args.weight_vol * loss_vol
        if "fwd_dd" in out and Args.weight_dd>0:
            tdd=torch.tensor(fdd, device=device)
            mask_dd = ~torch.isnan(tdd)
            if mask_dd.sum()>0:
                loss_dd = F.mse_loss(out["fwd_dd"][mask_dd], tdd[mask_dd])
                loss = loss + Args.weight_dd * loss_dd
                if Args.weight_var>0:
                    std_pred_dd = torch.std(out["fwd_dd"][mask_dd])
                    std_true_dd = torch.std(tdd[mask_dd])
                    var_loss_dd = (std_pred_dd - std_true_dd).pow(2)
                    loss = loss + Args.weight_var * var_loss_dd
        loss.backward()
        opt.step()
        total_loss+=float(loss.detach())
        total_nce+=float(nce_loss.detach())
        nb+=1
    avg_loss = total_loss / max(1,nb)
    avg_nce = total_nce / max(1,nb)
    print(f"Ep {epoch+1}/{Args.epochs} loss {avg_loss:.4f} nce {avg_nce:.4f} fwd {total_fwd/max(1,nb):.4f} entry {total_entry/max(1,nb):.4f}")
    if (epoch+1) % Args.val_every ==0:
        model.eval()
        pf_all=[]; tf_all=[]; pd_all=[]; td_all=[]; entry_logits=[]; entry_true=[]
        with torch.no_grad():
            for s in range(0, len(val_seqs), Args.batch):
                bl=val_seqs[s:s+Args.batch]
                xs_seq, ms_seq, te_seq, yn_seq, vm, sector_ids, f6,fvol,fdd,triple = build_batch(bl)
                xs_t={fam: torch.tensor(xs_seq[fam], device=device) for fam in fams}
                ms_t={fam: torch.tensor(ms_seq[fam], device=device) for fam in fams}
                te_t=torch.tensor(te_seq, device=device)
                yn_t=torch.tensor(yn_seq, device=device)
                vm_t=torch.tensor(vm, device=device)
                c_seq,z_seq,out = model.forward_sequence(xs_t, ms_t, te_t, yn_t, vm_t)
                pf = out["fwd_ret"][:,:,2].cpu().numpy()
                pd = out["fwd_dd"].cpu().numpy()
                for b in range(len(bl)):
                    for l in range(bl[b]["valid_len"]):
                        if not np.isnan(f6[b,l]):
                            pf_all.append(pf[b,l]); tf_all.append(f6[b,l])
                        if not np.isnan(fdd[b,l]):
                            pd_all.append(pd[b,l]); td_all.append(fdd[b,l])
                        if triple[b,l]!=-1:
                            entry_logits.append(out["entry"][b,l].item())
                            entry_true.append(triple[b,l])
        import scipy.stats as stats
        if len(pf_all)>10:
            ic_f6 = stats.spearmanr(pf_all, tf_all)[0]
            ic_dd = stats.spearmanr(pd_all, td_all)[0]
            entry_arr = np.array(entry_logits)
            true_arr = np.array(entry_true)
            order = np.argsort(1/(1+np.exp(-entry_arr)))[::-1]
            top20 = true_arr[order[:20]]
            prec20 = (top20==1).mean() if len(top20)>0 else 0
            print(f"  VAL f6 IC {ic_f6:.4f} dd IC {ic_dd:.4f} entry prec20 {prec20:.3f} mean pf {np.mean(pf_all):.3f} std {np.std(pf_all):.3f} mean pd {np.mean(pd_all):.3f} std {np.std(pd_all):.3f}")
            if ic_f6>best_ic:
                best_ic=ic_f6
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                torch.save({"model":best_state, "args":vars(Args), "epoch":epoch+1, "ic":ic_f6, "prec20":prec20}, Args.out)
                print(f"  -> saved best to {Args.out} IC {ic_f6:.4f}")
        model.train()
final_path = DATA_DIR/"mtnn_career_v6_last.pt"
torch.save({"model": model.state_dict(), "args": vars(Args)}, final_path)
print(f"Saved final to {final_path}")
