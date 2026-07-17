"""
Tune entry head with pos_weight / focal sweep, freezing backbone
"""
import torch, sys, numpy as np
from pathlib import Path
sys.path.insert(0,"pipeline")
from dataset_career import load_bundle, family_slices, build_sequences, get_time_enc_for_seq
from model_career import EquitiesCareerMTNN
import torch.nn.functional as F
from torch.utils.data import DataLoader

DATA_DIR=Path("pipeline/data")
Z, mask, Z_raw, tickers_b, names, fy_arr, sectors_arr, manifest, fwd, _ = load_bundle()
fams, feat_list = family_slices(manifest)
feat_to_idx={f:i for i,f in enumerate(feat_list)}
fam_dims={fam: len(cols) for fam, cols in fams.items()}
seqs,_,_ = build_sequences(Z, mask, Z_raw, tickers_b, fy_arr, sectors_arr, manifest, fwd)

device="cpu"
ckpt_path=DATA_DIR/"mtnn_career_best.pt"
ckpt=torch.load(ckpt_path, map_location=device, weights_only=False)
args=ckpt.get("args",{})
print(f"Loaded best checkpoint args {args} ic {ckpt.get('ic')} prec {ckpt.get('prec20')}")

def build_model():
    model=EquitiesCareerMTNN(
        fam_dims=fam_dims,
        d_tower=args.get("d_tower",24),
        d_tower_hidden=args.get("d_tower_hidden",64),
        d_emb=args.get("dim",64),
        d_time=8, d_time_emb=16,
        d_model=args.get("d_model",64),
        n_layers=args.get("n_layers",3),
        n_heads=args.get("n_heads",4),
        n_game=14, n_skills=0,
        n_sectors=11, n_archetypes=8, mlp_heads=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model

def build_batch(batch_seq_list):
    B=len(batch_seq_list); L=10
    xs_seq={fam: np.zeros((B,L,fam_dims[fam]),dtype=np.float32) for fam in fams}
    ms_seq={fam: np.zeros((B,L,fam_dims[fam]),dtype=np.float32) for fam in fams}
    time_enc_seq=np.zeros((B,L,8),dtype=np.float32)
    year_norm_seq=np.zeros((B,L,1),dtype=np.float32)
    valid_mask=np.zeros((B,L),dtype=bool)
    triple_target=np.full((B,L), -1, dtype=np.int32)
    for b,seq in enumerate(batch_seq_list):
        enc=get_time_enc_for_seq(seq, Z_raw, fwd, feat_to_idx)
        time_enc_seq[b]=enc["time_enc"]
        year_norm_seq[b]=enc["year_norm"]
        valid_mask[b]=enc["mask"]
        # triple barrier from fwd
        for pos, orig_idx in enumerate(seq["indices"]):
            if pos>=L: break
            # find corresponding fwd triple for this original idx? seq already has mapping
            # Use fwd['triple_barrier'] aligned to Z index via original matrix order?
            # Simpler: get from seq's fiscal year lookup via global fwd arrays indexed by orig_idx
            # fwd arrays are in order of Z (2741)
            triple_target[b,pos]=int(fwd['triple_barrier'][orig_idx])
        for pos, orig_idx in enumerate(seq["indices"]):
            if pos>=L: break
            for fam,cols in fams.items():
                xs_seq[fam][b,pos,:]=Z[orig_idx, cols]
                ms_seq[fam][b,pos,:]=mask[orig_idx, cols]
    return xs_seq, ms_seq, time_enc_seq, year_norm_seq, valid_mask, triple_target, batch_seq_list

def evaluate(model):
    model.eval()
    preds=[]
    trues=[]
    with torch.no_grad():
        for s in range(0,len(seqs),32):
            batch_list=seqs[s:s+32]
            xs_seq,ms_seq,te_seq,yn_seq,vm,triple,_=build_batch(batch_list)
            xs_t={fam: torch.tensor(xs_seq[fam],device=device) for fam in fams}
            ms_t={fam: torch.tensor(ms_seq[fam],device=device) for fam in fams}
            te_t=torch.tensor(te_seq,device=device)
            yn_t=torch.tensor(yn_seq,device=device)
            vm_t=torch.tensor(vm,device=device)
            c_seq,z_seq,out=model.forward_sequence(xs_t,ms_t,te_t,yn_t,vm_t)
            entry=torch.sigmoid(out["entry"]).cpu().numpy()
            mask_e=(triple!=-1)&vm
            preds.append(entry[mask_e])
            trues.append((triple[mask_e]==1).astype(int))
    if not preds: return 0,0,0,0
    p=np.concatenate(preds); t=np.concatenate(trues)
    # prec@20
    idx=np.argsort(p)[-20:]
    prec = t[idx].mean() if len(idx)>0 else 0
    # spread
    spread = p.max()-p.min()
    std = p.std()
    # brier
    return p.mean(), spread, std, prec

# Try different configs
configs=[
    {"name":"bce_pw1.0_w3.0_gamma0", "pos_weight":1.0, "gamma":0.0, "w":3.0, "lr":0.005, "epochs":12},
    {"name":"bce_pw2.0_w3.0_gamma0", "pos_weight":2.0, "gamma":0.0, "w":3.0, "lr":0.005, "epochs":12},
    {"name":"focal_pw2.0_gamma2_w3", "pos_weight":2.0, "gamma":2.0, "w":3.0, "lr":0.003, "epochs":12},
    {"name":"focal_pw3.0_gamma1_w4", "pos_weight":3.0, "gamma":1.0, "w":4.0, "lr":0.003, "epochs":12},
    {"name":"bce_pw1.5_w5_gamma0_deeperHead", "pos_weight":1.5, "gamma":0.0, "w":5.0, "lr":0.005, "epochs":15, "deeper":True},
]

for cfg in configs:
    print(f"\n=== Tuning {cfg['name']} ===")
    model=build_model()
    # optionally make entry head deeper
    if cfg.get("deeper"):
        import torch.nn as nn
        d_emb=args.get("dim",64)
        model.entry_head = nn.Sequential(nn.Linear(d_emb, 32), nn.ReLU(), nn.Linear(32,1)).to(device)
        # re-init deeper
    # freeze all except entry_head
    for name,param in model.named_parameters():
        param.requires_grad = "entry_head" in name
    # check trainable count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params {trainable}")
    opt=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    best_prec=0
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss=0
        n_batches=0
        for s in range(0,len(seqs),16):
            batch_list=seqs[s:s+16]
            xs_seq,ms_seq,te_seq,yn_seq,vm,triple,_=build_batch(batch_list)
            xs_t={fam: torch.tensor(xs_seq[fam],device=device) for fam in fams}
            ms_t={fam: torch.tensor(ms_seq[fam],device=device) for fam in fams}
            te_t=torch.tensor(te_seq,device=device)
            yn_t=torch.tensor(yn_seq,device=device)
            vm_t=torch.tensor(vm,device=device)
            triple_t=torch.tensor(triple,device=device)
            opt.zero_grad()
            c_seq,z_seq,out=model.forward_sequence(xs_t,ms_t,te_t,yn_t,vm_t)
            logits=out["entry"]
            mask_t = (triple_t!=-1) & vm_t
            if not mask_t.any():
                continue
            target = (triple_t[mask_t]==1).float()
            logit = logits[mask_t]
            pw=torch.tensor([cfg["pos_weight"]], device=device)
            if cfg["gamma"]==0:
                loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)
            else:
                bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw, reduction='none')
                prob=torch.sigmoid(logit)
                pt=torch.where(target==1, prob, 1-prob)
                focal_w=(1-pt).pow(cfg["gamma"])
                loss=(focal_w*bce).mean()
            loss = loss * cfg["w"]
            loss.backward()
            opt.step()
            total_loss+=loss.item()
            n_batches+=1
        mean, spread, std, prec = evaluate(model)
        print(f"Ep {epoch+1}/{cfg['epochs']} loss {total_loss/max(1,n_batches):.4f} -> entry mean {mean:.3f} spread {spread:.3f} std {std:.4f} prec@20 {prec:.3f}")
        if prec>best_prec:
            best_prec=prec
            # save
            torch.save({"model":model.state_dict(), "cfg":cfg, "prec":prec, "mean":mean, "spread":spread}, f"pipeline/data/mtnn_entry_tuned_{cfg['name']}.pt")
    print(f"Best prec for {cfg['name']}: {best_prec}")
