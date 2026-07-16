"""
Vector Equities MTNN Training — cloned from vector-hoops train_mtnn.py v4 Phase B

Train 48-d company embedding from holistic SEC + NEO + market + ownership + text families.

Multi-task: InfoNCE same-ticker adjacent FY + archetype 8 + sector 11 + next FY profile + 12 skill towers + valuation + market + health etc.

Usage:
 python pipeline/train_mtnn.py --epochs 40 --dim 48 --fusion gated

Outputs: pipeline/data/embedding.npz + mtnn_best.pt + mtnn_report.json
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
import sys
sys.path.insert(0, str(ROOT / "pipeline"))
from model import EquitiesMTNN
from composite_score import composite_quality

# Loss weights — mirrors hoops Phase B rebalanced
DEFAULT_WEIGHTS = {
    "archetype": 0.25,
    "sector": 0.15,
    "profile": 0.12,
    "next_profile": 0.10,
    "skills": 0.20,
    "valuation": 0.12,
    "market": 0.12,
    "vol": 0.05,
    "health": 0.08,
    "payout": 0.05,
    "mgmt": 0.08,
    "own": 0.05,
}

def load_bundle():
    npz = np.load(DATA_DIR / "train_matrix.npz", allow_pickle=False)
    manifest = json.loads((DATA_DIR / "feature_manifest.json").read_text())
    Z = npz["Z"].astype(np.float32)
    mask = npz["mask"].astype(np.float32)
    tickers = npz["ticker"].astype(str)
    names = npz["name"].astype(str)
    fiscal_years = npz["fiscal_year"].astype(str)
    sectors = npz["sector"].astype(str)
    clusters = npz["cluster"].astype(np.int64) if "cluster" in npz else np.zeros(len(Z), dtype=np.int64)
    return Z, mask, tickers, names, fiscal_years, sectors, clusters, manifest

def family_slices(manifest):
    fams = defaultdict(list)
    for j,f in enumerate(manifest["features"]):
        fam = manifest["families"][j]
        fams[fam].append(j)
    return dict(fams)

def season_index(fiscal_years):
    uniq = sorted(set(str(s) for s in fiscal_years))
    m = {s:i for i,s in enumerate(uniq)}
    return np.array([m[str(s)] for s in fiscal_years], dtype=np.int64), uniq

def adjacent_pairs(tickers, fiscal_years):
    def y(s): return int(s[:4])
    by = defaultdict(list)
    for i,(t,s) in enumerate(zip(tickers, fiscal_years)):
        by[t].append((y(str(s)), i))
    pairs=[]
    for rows in by.values():
        rows.sort()
        for (y1,i1),(y2,i2) in zip(rows, rows[1:]):
            if y2-y1==1:
                pairs.append((i1,i2))
    return pairs

def next_index(n, pairs):
    nxt = np.full(n, -1, dtype=np.int64)
    for a,b in pairs:
        nxt[a]=b
    return nxt

def load_skills(names, fiscal_years):
    path = DATA_DIR / "skill_labels.npz"
    if not path.exists():
        return np.zeros((len(names),0), np.float32), np.zeros((len(names),0), np.float32), []
    npz = np.load(path, allow_pickle=False)
    keys = [str(k) for k in npz["keys"]]
    lookup = {(str(n), str(s)): g for n,s,g in zip(npz["name"], npz["season"], npz["grades"])}
    G = np.zeros((len(names), len(keys)), dtype=np.float32)
    M = np.zeros((len(names), len(keys)), dtype=np.float32)
    for i,(n,s) in enumerate(zip(names, fiscal_years)):
        g = lookup.get((str(n), str(s)))
        if g is not None:
            G[i]=g/100.0  # normalize 0-1 for training
            M[i]=1.0
    return G, M, keys

def split_by_family(Z,M,fams,device):
    xs,ms={},{}
    for fam,cols in fams.items():
        xs[fam]=torch.tensor(Z[:,cols], device=device)
        ms[fam]=torch.tensor(M[:,cols], device=device)
    return xs,ms

def batch_views(xs, ms, idx, drop_p=0.12):
    out_x,out_m={},{}
    for fam in xs:
        x=xs[fam][idx]; m=ms[fam][idx]
        keep=(torch.rand_like(m)>drop_p).float()
        out_x[fam]=x*keep
        out_m[fam]=m*keep
    return out_x,out_m

def game_cols(manifest):
    game = manifest.get("game_features", [])
    return [manifest["features"].index(f) for f in game if f in manifest["features"]]

def tensor_col(Z,M,j,device):
    return torch.tensor(Z[:,j], device=device), torch.tensor(M[:,j], device=device)

def masked_scalar_mse(pred, target, row_mask):
    w=row_mask
    if w.sum()<=0: return pred.sum()*0.0
    return (w*(pred-target)**2).sum()/w.sum()

def info_nce(za,zb,temp=0.08, pos_a=None, pos_b=None, hard_boost=0.0):
    logits = za @ zb.T / temp
    if hard_boost>0 and pos_a is not None and pos_b is not None:
        b=logits.shape[0]
        idx=torch.arange(b, device=logits.device)
        hard=(pos_a.unsqueeze(1)==pos_b.unsqueeze(0)) & (idx.unsqueeze(0)!=idx.unsqueeze(1))
        logits=logits + hard.float()*hard_boost
    target=torch.arange(len(za), device=za.device)
    return 0.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))

def recall_at_k(E, pairs, k=10):
    if len(pairs)==0: return None
    sample=pairs[np.random.choice(len(pairs), min(500,len(pairs)), replace=False)] if len(pairs)>500 else pairs
    hits=0
    for a,b in sample:
        sims=E @ E[a]
        sims[a]=-np.inf
        top=np.argpartition(-sims,k)[:k]
        hits+=int(b in top)
    return hits/len(sample)

def cross_cycle_purity(E, clusters, fiscal_years, k=20, n_sample=400):
    years=np.array([int(str(s)[:4]) for s in fiscal_years])
    rng=np.random.default_rng(7)
    candidates=np.where(clusters>=0)[0]
    if len(candidates)<n_sample: return None
    sample=rng.choice(candidates, min(n_sample,len(candidates)), replace=False)
    pur=[]
    for i in sample:
        sims=E @ E[i]; sims[i]=-np.inf
        top=np.argpartition(-sims,k)[:k]
        # cross-cycle = different year != same year
        cross=top[years[top]!=years[i]]
        if len(cross)==0: continue
        pur.append(float((clusters[cross]==clusters[i]).mean()))
    return float(np.mean(pur)) if pur else None

def eval_split(fy_str):
    y=int(str(fy_str)[:4])
    if y<=2021: return "train"
    if y<=2023: return "val"
    return "test"

def filter_pairs_by_split(pairs, fiscal_years, split):
    keep=[]
    for a,b in pairs:
        if eval_split(str(fiscal_years[b]))==split:
            keep.append((int(a),int(b)))
    return np.array(keep,dtype=int) if keep else np.zeros((0,2),int)

@torch.no_grad()
def embed_all(model, xs, ms, seas_t):
    model.eval()
    return model.encode(xs, ms, seas_t).cpu().numpy().astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--tower-width", type=int, default=24)
    ap.add_argument("--tower-hidden", type=int, default=96)
    ap.add_argument("--skill-hidden", type=int, default=16)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nce-temp", type=float, default=0.08)
    ap.add_argument("--drop-p", type=float, default=0.12)
    ap.add_argument("--hard-neg-boost", type=float, default=0.2)
    ap.add_argument("--fusion", choices=("gated","concat","transformer"), default="gated")
    ap.add_argument("--tower-blocks", type=int, default=1)
    ap.add_argument("--mlp-heads", action="store_true")
    ap.add_argument("--d-head-hidden", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-fusion-layers", type=int, default=4)
    ap.add_argument("--n-attn-heads", type=int, default=4)
    ap.add_argument("--fusion-hidden", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=5)
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"

    Z, mask, tickers, names, fiscal_years, sectors, clusters, manifest = load_bundle()
    fams = family_slices(manifest)
    season_ids_arr, uniq_years = season_index(fiscal_years)
    n_seasons = len(uniq_years)
    gcols = game_cols(manifest)
    print(f"{len(Z)} rows, {Z.shape[1]} feats, {len(fams)} towers, {n_seasons} FYs, device={device}")
    print(f"families: { {k:len(v) for k,v in fams.items()} }")
    print(f"game profile cols {gcols} -> {len(gcols)} dims")

    # sector mapping
    from feature_spec import SECTORS
    sector_to_idx = {s:i for i,s in enumerate(SECTORS)}
    sector_idx_arr = np.array([sector_to_idx.get(s, -1) for s in sectors], dtype=np.int64)
    n_sectors = len(SECTORS)

    pairs = adjacent_pairs(tickers, fiscal_years)
    print(f"{len(pairs)} adjacent FY pairs")
    pair_arr = np.array(pairs) if pairs else np.zeros((0,2), int)
    next_idx_arr = next_index(len(Z), pairs)
    next_count = int((next_idx_arr>=0).sum())
    print(f"next-FY labels {next_count}/{len(Z)}")

    # Skills
    skill_g, skill_m, skill_keys = load_skills(names, fiscal_years)
    print(f"{len(skill_keys)} skills")

    # Tensors
    game_z = torch.tensor(Z[:, gcols], device=device) if gcols else None
    # sector tensor
    sector_t = torch.tensor(sector_idx_arr, device=device)
    arch_t = torch.tensor(clusters, device=device)
    seas_t = torch.tensor(season_ids_arr, device=device)
    # valuation proxy = EV_EBITDA z
    ev_idx = manifest["features"].index("EV_EBITDA") if "EV_EBITDA" in manifest["features"] else None
    ev_z, ev_m = tensor_col(Z, mask, ev_idx, device) if ev_idx is not None else (None,None)
    ret_idx = manifest["features"].index("RET_12M") if "RET_12M" in manifest["features"] else None
    ret_z, ret_m = tensor_col(Z, mask, ret_idx, device) if ret_idx is not None else (None,None)
    alt_idx = manifest["features"].index("ALTMAN_Z") if "ALTMAN_Z" in manifest["features"] else None
    # fallback: use roe as health proxy if altman not present
    if alt_idx is None:
        alt_idx = manifest["features"].index("ROE") if "ROE" in manifest["features"] else None
    health_z, health_m = tensor_col(Z, mask, alt_idx, device) if alt_idx is not None else (None,None)

    skill_t = torch.tensor(skill_g, device=device)
    skillm_t = torch.tensor(skill_m, device=device)

    # split
    split_of = np.array([eval_split(str(s)) for s in fiscal_years])
    fit_mask = split_of=="train"
    fit_idx = np.where(fit_mask)[0]
    print(f"fit rows train {len(fit_idx)}/{len(Z)} holdout val {int((split_of=='val').sum())} test {int((split_of=='test').sum())}")

    # model
    xs, ms = split_by_family(Z, mask, fams, device)
    model = EquitiesMTNN(
        {f:len(c) for f,c in fams.items()}, n_seasons,
        d_tower=args.tower_width, d_tower_hidden=args.tower_hidden, d_emb=args.dim,
        n_game=len(gcols), n_skills=len(skill_keys),
        d_skill_hidden=args.skill_hidden,
        n_sectors=n_sectors, n_archetypes=8,
        fusion_mode=args.fusion, n_tower_blocks=args.tower_blocks,
        mlp_heads=args.mlp_heads, d_head_hidden=args.d_head_hidden,
        d_model=args.d_model, n_fusion_layers=args.n_fusion_layers,
        n_attn_heads=args.n_attn_heads,
        d_fusion_hidden=(args.fusion_hidden or None)
    ).to(device)

    # adamw no-decay on biases + LN
    decay, no_decay=[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim==1 or name.endswith(".bias"): no_decay.append(p)
        else: decay.append(p)
    opt = torch.optim.AdamW([{"params":decay,"weight_decay":args.weight_decay},{"params":no_decay,"weight_decay":0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1,(len(fit_idx)//args.batch+1)*args.epochs), pct_start=0.1, anneal_strategy="linear")

    # training
    lookup = {}
    if len(pair_arr):
        lookup={int(a):int(b) for a,b in pair_arr}
        lookup.update({int(b):int(a) for a,b in pair_arr})

    best_val_recall=None; best_epoch=-1; best_cqs=None; best_composite=-1
    BEST_CKPT=DATA_DIR / "mtnn_best.pt"

    history=[]
    for epoch in range(args.epochs):
        model.train()
        perm=np.random.permutation(fit_idx)
        total=0.0; steps=0; accum=0
        opt.zero_grad(set_to_none=True)
        for s in range(0,len(perm),args.batch):
            idx=perm[s:s+args.batch]
            if len(idx)<8: continue
            idx_t=torch.tensor(idx, device=device)
            partner=np.array([lookup.get(int(i), int(i)) for i in idx])
            partner_t=torch.tensor(partner, device=device)

            xa,ma=batch_views(xs,ms,idx_t,drop_p=args.drop_p)
            xb,mb=batch_views(xs,ms,partner_t,drop_p=args.drop_p)
            za,out_a=model(xa,ma,seas_t[idx_t])
            zb,_=model(xb,mb,seas_t[partner_t])

            loss=info_nce(za,zb,temp=args.nce_temp, pos_a=sector_t[idx_t], pos_b=sector_t[partner_t], hard_boost=args.hard_neg_boost)
            loss=loss + DEFAULT_WEIGHTS["archetype"]*F.cross_entropy(out_a["archetype"], arch_t[idx_t])
            valid_sec = sector_t[idx_t]>=0
            if valid_sec.any():
                loss=loss + DEFAULT_WEIGHTS["sector"]*F.cross_entropy(out_a["sector"][valid_sec], sector_t[idx_t][valid_sec])
            if game_z is not None:
                loss=loss + DEFAULT_WEIGHTS["profile"]*F.mse_loss(out_a["profile"], game_z[idx_t])
                # next profile
                nxt_batch=next_idx_arr[idx]
                valid_next=nxt_batch>=0
                if valid_next.any():
                    nxt_t=torch.tensor(nxt_batch[valid_next], device=device)
                    valid_t=torch.tensor(valid_next, device=device)
                    loss=loss + DEFAULT_WEIGHTS["next_profile"]*F.smooth_l1_loss(out_a["next_profile"][valid_t], game_z[nxt_t])

            if "skills" in out_a and skill_t.shape[1]>0:
                wm=skillm_t[idx_t]
                if wm.sum()>0:
                    se=(out_a["skills"]-skill_t[idx_t])**2
                    loss=loss + DEFAULT_WEIGHTS["skills"]*(wm*se).sum()/wm.sum()

            if ev_z is not None:
                loss=loss + DEFAULT_WEIGHTS["valuation"]*masked_scalar_mse(out_a["valuation"], ev_z[idx_t], ev_m[idx_t])
            if ret_z is not None:
                loss=loss + DEFAULT_WEIGHTS["market"]*masked_scalar_mse(out_a["market"], ret_z[idx_t], ret_m[idx_t])
            if health_z is not None:
                loss=loss + DEFAULT_WEIGHTS["health"]*masked_scalar_mse(out_a["health"], health_z[idx_t], health_m[idx_t])

            scaled=loss/args.grad_accum
            scaled.backward()
            accum+=1; total+=float(loss)
            if accum<args.grad_accum: continue
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step()
            opt.zero_grad(set_to_none=True)
            accum=0; steps+=1
        if accum>0:
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step()
            opt.zero_grad(set_to_none=True)
            steps+=1
        avg=total/max(1,steps)
        history.append(avg)

        if (epoch%args.val_every==0 or epoch==args.epochs-1):
            E_val=embed_all(model, xs, ms, seas_t)
            val_pairs=filter_pairs_by_split(pair_arr, fiscal_years, "val")
            test_pairs=filter_pairs_by_split(pair_arr, fiscal_years, "test")
            val_r=recall_at_k(E_val, val_pairs, k=10)
            test_r=recall_at_k(E_val, test_pairs, k=10)
            val_pur=cross_cycle_purity(E_val, clusters, fiscal_years)
            # composite proxy like hoops partial_cqs
            comp_proxy = (0.5*(val_r or 0) + 0.5*(val_pur or 0)) if val_r is not None else 0
            print(f"epoch {epoch:3d} loss {avg:.4f} val_recall@10={val_r} test_recall@10={test_r} purity={val_pur} comp={comp_proxy:.3f} lr {sched.get_last_lr()[0]:.2e}")
            # checkpoint on composite proxy (better than recall-only which picks epoch 0)
            if comp_proxy is not None and comp_proxy>best_composite:
                best_composite=comp_proxy; best_val_recall=val_r; best_epoch=epoch; best_cqs=comp_proxy
                torch.save({"epoch":epoch,"model":model.state_dict(),"val_recall":val_r,"val_purity":val_pur,"composite":comp_proxy,"args":vars(args)}, BEST_CKPT)
        else:
            if epoch%5==0:
                print(f"epoch {epoch:3d} loss {avg:.4f}")

    # restore best
    if BEST_CKPT.exists():
        ckpt=torch.load(BEST_CKPT, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"restored best epoch {ckpt['epoch']} recall {ckpt.get('val_recall')} purity {ckpt.get('val_purity')} comp {ckpt.get('composite')}")

    # final export
    model.eval()
    with torch.no_grad():
        emb=model.encode(xs, ms, seas_t)
        _, heads=model(xs, ms, seas_t)
        tower_stack=torch.stack([model.towers[fam](xs[fam], ms[fam]) for fam in fams], dim=1)

    E=emb.cpu().numpy().astype(np.float32)
    arch_logits=heads["archetype"].cpu().numpy().astype(np.float32)
    sector_logits=heads["sector"].cpu().numpy().astype(np.float32)
    skill_pred=heads["skills"].cpu().numpy().astype(np.float32) if "skills" in heads else np.zeros((len(E),0), np.float32)
    next_pred=heads["next_profile"].cpu().numpy().astype(np.float32) if "next_profile" in heads else np.zeros((len(E),len(gcols)), np.float32)

    np.savez_compressed(DATA_DIR / "embedding.npz", E=E, ticker=tickers, name=names, fiscal_year=fiscal_years, sector=sectors, cluster=clusters,
                        archetype_logits=arch_logits, sector_logits=sector_logits, skill_pred=skill_pred, skill_keys=np.array(skill_keys),
                        next_profile_pred=next_pred)

    # metrics
    held={}
    for split in ("train","val","test","all"):
        sub=pair_arr if split=="all" else filter_pairs_by_split(pair_arr, fiscal_years, split)
        held[split]={"pairs":int(len(sub)),"recall_at_10_mtnn":recall_at_k(E, sub, k=10)}

    # next profile r2 per split
    next_report={}
    for split in ("val","test"):
        rows=np.where((next_idx_arr>=0) & (split_of==split))[0] if split!="all" else np.where(next_idx_arr>=0)[0]
        # but we need target split = target fy split, not source? use filter_pairs
        sub=filter_pairs_by_split(pair_arr, fiscal_years, split)
        if len(sub)==0:
            next_report[split]=None; continue
        a=sub[:,0]; b=sub[:,1]
        y=Z[b][:,gcols] if gcols else np.zeros((len(b),0))
        p=next_pred[a]
        resid=y-p
        mse=float((resid**2).mean()) if len(resid) else 0
        rmse=float(np.sqrt(mse)) if mse else 0
        # R2
        ss_tot=float(((y - y.mean(axis=0,keepdims=True))**2).sum()) if len(y) else 1
        r2=1.0 - float((resid**2).sum())/max(ss_tot,1e-9)
        next_report[split]={"rows":int(len(sub)),"mae_z":float(np.abs(resid).mean()) if len(resid) else None,"r2":round(r2,4),"rmse":round(rmse,4)}

    purity=cross_cycle_purity(E, clusters, fiscal_years)

    # sector acc
    sector_pred=sector_logits.argmax(1)
    valid=sector_idx_arr>=0
    sector_acc=float((sector_pred[valid]==sector_idx_arr[valid]).mean()) if valid.sum() else None

    # market directional acc (proxy: sign of predicted vs actual ret)
    market_acc=None
    if ret_idx is not None and "market" in heads:
        actual=Z[:,ret_idx]
        pred_m=heads["market"].cpu().numpy()
        # sign agreement
        mask=mask[:,ret_idx]>0
        if mask.sum()>10:
            market_acc=float(((np.sign(actual[mask])==np.sign(pred_m[mask])).mean()))

    report={
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "model": f"equities_mtnn_v4_{args.fusion}_d{args.dim}_b{args.tower_blocks}",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_recall_at_10": best_val_recall,
        "dim": args.dim,
        "fusion": args.fusion,
        "tower_blocks": args.tower_blocks,
        "towers": {k:len(v) for k,v in fams.items()},
        "positive_pairs": len(pairs),
        "final_loss": history[-1] if history else None,
        "held_out_recall": held,
        "cross_cycle_archetype_purity_at_20": purity,
        "sector_top1_acc": sector_acc,
        "next_profile": next_report,
        "market_directional_acc": market_acc,
        "loss_weights": DEFAULT_WEIGHTS,
    }
    report["composite"]=composite_quality(report)
    ok, why = (report["composite"]["cqs"]>=0.6, f"CQS {report['composite']['cqs']}")
    # promote gate simplified
    report["promote"]={"ok": float(report["composite"]["cqs"])>=0.6, "reason": f"CQS {report['composite']['cqs']}"}

    (DATA_DIR / "mtnn_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"CQS {report['composite']['cqs']} — {report['promote']}")

if __name__=="__main__":
    main()
