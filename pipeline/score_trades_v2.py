"""
Scoring v2 - adaptive thresholds
"""
import torch, sys, json, numpy as np
from pathlib import Path
sys.path.insert(0,"pipeline")
from dataset_career import load_bundle, family_slices, build_sequences, get_time_enc_for_seq
from model_career import EquitiesCareerMTNN
from collections import defaultdict

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
model=EquitiesCareerMTNN(
    fam_dims=fam_dims,
    d_tower=args.get("d_tower",12),
    d_tower_hidden=args.get("d_tower_hidden",48),
    d_emb=args.get("dim",32),
    d_time=8, d_time_emb=16,
    d_model=args.get("d_model",32),
    n_layers=args.get("n_layers",2),
    n_heads=args.get("n_heads",2),
    n_game=14, n_skills=0,
    n_sectors=11, n_archetypes=8, mlp_heads=True,
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

def build_batch(batch_seq_list):
    B=len(batch_seq_list); L=10
    xs_seq={fam: np.zeros((B,L,fam_dims[fam]),dtype=np.float32) for fam in fams}
    ms_seq={fam: np.zeros((B,L,fam_dims[fam]),dtype=np.float32) for fam in fams}
    time_enc_seq=np.zeros((B,L,8),dtype=np.float32)
    year_norm_seq=np.zeros((B,L,1),dtype=np.float32)
    valid_mask=np.zeros((B,L),dtype=bool)
    for b,seq in enumerate(batch_seq_list):
        enc=get_time_enc_for_seq(seq, Z_raw, fwd, feat_to_idx)
        time_enc_seq[b]=enc["time_enc"]
        year_norm_seq[b]=enc["year_norm"]
        valid_mask[b]=enc["mask"]
        for pos,orig_idx in enumerate(seq["indices"]):
            if pos>=L: break
            for fam,cols in fams.items():
                xs_seq[fam][b,pos,:]=Z[orig_idx, cols]
                ms_seq[fam][b,pos,:]=mask[orig_idx, cols]
    return xs_seq, ms_seq, time_enc_seq, year_norm_seq, valid_mask, batch_seq_list

all_preds=[]
for s in range(0,len(seqs),32):
    batch_list=seqs[s:s+32]
    xs_seq,ms_seq,te_seq,yn_seq,vm,_=build_batch(batch_list)
    xs_t={fam: torch.tensor(xs_seq[fam],device=device) for fam in fams}
    ms_t={fam: torch.tensor(ms_seq[fam],device=device) for fam in fams}
    te_t=torch.tensor(te_seq,device=device)
    yn_t=torch.tensor(yn_seq,device=device)
    vm_t=torch.tensor(vm,device=device)
    with torch.no_grad():
        c_seq,z_seq,out=model.forward_sequence(xs_t,ms_t,te_t,yn_t,vm_t)
        fwd_pred=out["fwd_ret"][:,:,2].cpu().numpy()
        entry=torch.sigmoid(out["entry"]).cpu().numpy()
        dd=out["fwd_dd"].cpu().numpy()
        fwd_vol=out["fwd_vol"].cpu().numpy() if "fwd_vol" in out else np.zeros_like(entry)
        for b_idx,seq in enumerate(batch_list):
            for l in range(seq["valid_len"]):
                fy=int(seq["fiscal_years"][l])
                t=seq["ticker"]
                all_preds.append({
                    "ticker":t,
                    "fy":fy,
                    "entry":float(entry[b_idx,l]),
                    "fwd6":float(fwd_pred[b_idx,l]),
                    "dd":float(dd[b_idx,l]),
                    "fwd_vol":float(fwd_vol[b_idx,l]),
                    "sector":seq["sector"]
                })

# latest per ticker
from collections import defaultdict
latest_map={}
for p in all_preds:
    key=p["ticker"]
    if key not in latest_map or p["fy"]>latest_map[key]["fy"]:
        latest_map[key]=p

latest=list(latest_map.values())
# sort by composite score entry * fwd6 - dd penalty
for p in latest:
    # penalize large drawdown
    p["score"]=p["entry"]*p["fwd6"] + p["dd"]*0.1  # dd negative, so add

# adaptive thresholds: use median
import numpy as np
entries=np.array([p["entry"] for p in latest])
fwd6s=np.array([p["fwd6"] for p in latest])
dds=np.array([p["dd"] for p in latest])
print(f"Entry mean {entries.mean():.3f} std {entries.std():.3f} min {entries.min():.3f} max {entries.max():.3f}")
print(f"Fwd6 mean {fwd6s.mean():.3f} std {fwd6s.std():.3f} min {fwd6s.min():.3f} max {fwd6s.max():.3f}")
print(f"DD mean {dds.mean():.3f} min {dds.min():.3f} max {dds.max():.3f}")

# Strict original filter
strict=[p for p in latest if p["entry"]>0.7 and p["fwd6"]>0.05 and p["dd"]>-0.10]
print(f"Strict filter 0.7/5%/-10% count {len(strict)}")

# Relaxed filters
relax1=[p for p in latest if p["entry"]>0.5 and p["fwd6"]>0.03 and p["dd"]>-0.15]
print(f"Relaxed 0.5/3%/-15% count {len(relax1)}")
relax2=[p for p in latest if p["fwd6"]>0.02 and p["dd"]>-0.20]
print(f"Relaxed fwd>2% dd>-20% count {len(relax2)}")

# Top 20 by score where fwd6>0.02
filtered=[p for p in latest if p["fwd6"]>0.02 and p["dd"]>-0.20]
top20=sorted(filtered, key=lambda x: x["score"], reverse=True)[:20]
print("Top20")
for i,p in enumerate(top20,1):
    print(f"{i}. {p['ticker']} FY{p['fy']} entry {p['entry']:.3f} fwd6 {p['fwd6']:.3f} dd {p['dd']:.3f} score {p['score']:.3f} sector {p['sector']}")

# Save
Path("pipeline/data/trades_career_relaxed.json").write_text(json.dumps(top20, indent=2))
md=["# Career MTNN Trades - Relaxed (adaptive)","",f"Model {ckpt_path} IC best {ckpt.get('ic')} prec {ckpt.get('prec20')}",f"Entry distrib mean {entries.mean():.3f} max {entries.max():.3f}",f"Strict 0.7/5%/-10% => {len(strict)} trades","Relaxed 0.5/3%/-15% => {len(relax1)}","Top20 fwd>2% dd>-20% sorted by entry*fwd6","", "| Rank | Ticker | FY | Entry | Fwd6M | DD | Score | Sector |","|---|---|---|---|---|---|---|---|"]
for i,p in enumerate(top20,1):
    md.append(f"| {i} | {p['ticker']} | {p['fy']} | {p['entry']:.3f} | {p['fwd6']:.3f} | {p['dd']:.3f} | {p['score']:.3f} | {p['sector']} |")
Path("pipeline/data/trades_career_relaxed.md").write_text("\n".join(md))
print("Saved relaxed")
