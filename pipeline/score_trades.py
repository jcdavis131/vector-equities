"""
Trading layer — continuous career MTNN
Filter: entry_score>0.7 & fwd_ret_6M>5% & dd<10%

Uses career model predictions or fallback heuristics
Outputs trade list JSON + markdown table

"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"


def load_bundle():
    candidates = [
        DATA_DIR / "train_matrix_v5.npz",
        DATA_DIR / "train_matrix_v4.npz",
        DATA_DIR / "train_matrix.npz",
    ]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        raise FileNotFoundError("No train matrix")
    npz = np.load(path, allow_pickle=True)
    print(f"Loaded {path} N={len(npz['ticker'])}")
    return npz, path


def load_career_embedding():
    cand = [DATA_DIR / "embedding_career.npz", DATA_DIR / "embedding.npz"]
    for c in cand:
        if c.exists():
            npz = np.load(c, allow_pickle=True)
            print(f"Loaded embedding {c} shape {npz['E'].shape}")
            return npz, c
    return None, None


def score_trades(
    entry_threshold=0.7, fwd_ret_threshold=0.05, dd_threshold=-0.10, top_k=50
):
    npz, _bundle_path = load_bundle()
    tickers = npz["ticker"].astype(str)
    fiscal_years = npz["fiscal_year"].astype(str)
    sectors = (
        npz["sector"].astype(str)
        if "sector" in npz
        else np.array(["Unknown"] * len(tickers))
    )
    price = npz["price"] if "price" in npz else np.zeros(len(tickers))
    fwd_ret_6m_true = (
        npz["fwd_ret_6m"] if "fwd_ret_6m" in npz else np.full(len(tickers), np.nan)
    )
    fwd_dd_6m_true = (
        npz["fwd_dd_6m"] if "fwd_dd_6m" in npz else np.full(len(tickers), np.nan)
    )
    triple_true = (
        npz["triple_barrier"] if "triple_barrier" in npz else np.full(len(tickers), -1)
    )
    # Try to load model predictions if exist
    # Load career model predictions from trained model? For now use heuristics or model if available
    entry_score = np.zeros(len(tickers), dtype=np.float32)
    fwd_ret_pred = np.full(len(tickers), np.nan, dtype=np.float32)
    fwd_dd_pred = np.full(len(tickers), np.nan, dtype=np.float32)

    # If we have trained career model, load its predictions from embedding? We'll try to load latest predictions from mtnn_career model if exists
    model_path = DATA_DIR / "mtnn_career_best.pt"
    if model_path.exists():
        try:
            import torch
            from dataset_career import (
                build_sequences,
                family_slices,
                get_time_enc_for_seq,
            )
            from dataset_career import load_bundle as lb2
            from model_career import EquitiesCareerMTNN

            Z, mask, Z_raw, tickers_b, _names, fy_arr, sectors_arr, manifest, fwd, _ = (
                lb2()
            )
            fams, feat_list = family_slices(manifest)
            feat_to_idx = {f: i for i, f in enumerate(feat_list)}
            fam_dims = {fam: len(cols) for fam, cols in fams.items()}
            seqs, _, _ = build_sequences(
                Z, mask, Z_raw, tickers_b, fy_arr, sectors_arr, manifest, fwd
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            args = ckpt.get("args", {})
            model = EquitiesCareerMTNN(
                fam_dims=fam_dims,
                d_tower=args.get("d_tower", 24),
                d_tower_hidden=args.get("d_tower_hidden", 96),
                d_emb=args.get("dim", 48),
                d_time=8,
                d_time_emb=16,
                d_model=args.get("d_model", 96),
                n_layers=args.get("n_layers", 4),
                n_heads=args.get("n_heads", 4),
                n_game=14,
                n_skills=0,
                n_sectors=len(
                    [
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
                    ]
                ),
                n_archetypes=8,
                mlp_heads=True,
            ).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            # Build map from (ticker,fy) -> pred
            pred_map = {}

            def build_batch(batch_seq_list):
                B = len(batch_seq_list)
                L = 10
                xs_seq = {
                    fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32)
                    for fam in fams
                }
                ms_seq = {
                    fam: np.zeros((B, L, fam_dims[fam]), dtype=np.float32)
                    for fam in fams
                }
                time_enc_seq = np.zeros((B, L, 8), dtype=np.float32)
                year_norm_seq = np.zeros((B, L, 1), dtype=np.float32)
                valid_mask = np.zeros((B, L), dtype=bool)
                for b, seq in enumerate(batch_seq_list):
                    enc = get_time_enc_for_seq(seq, Z_raw, fwd, feat_to_idx)
                    time_enc_seq[b] = enc["time_enc"]
                    year_norm_seq[b] = enc["year_norm"]
                    valid_mask[b] = enc["mask"]
                    for pos, orig_idx in enumerate(seq["indices"]):
                        if pos >= L:
                            break
                        for fam, cols in fams.items():
                            xs_seq[fam][b, pos, :] = Z[orig_idx, cols]
                            ms_seq[fam][b, pos, :] = mask[orig_idx, cols]
                return (
                    xs_seq,
                    ms_seq,
                    time_enc_seq,
                    year_norm_seq,
                    valid_mask,
                    batch_seq_list,
                )

            with torch.no_grad():
                for s in range(0, len(seqs), 32):
                    batch_list = seqs[s : s + 32]
                    xs_seq, ms_seq, time_enc_seq, year_norm_seq, valid_mask, blist = (
                        build_batch(batch_list)
                    )
                    xs_t = {
                        fam: torch.tensor(xs_seq[fam], device=device) for fam in fams
                    }
                    ms_t = {
                        fam: torch.tensor(ms_seq[fam], device=device) for fam in fams
                    }
                    te_t = torch.tensor(time_enc_seq, device=device)
                    yn_t = torch.tensor(year_norm_seq, device=device)
                    vm_t = torch.tensor(valid_mask, device=device)
                    _c_seq, _z_seq, out = model.forward_sequence(
                        xs_t, ms_t, te_t, yn_t, vm_t
                    )
                    fwd_pred = out["fwd_ret"][:, :, 2].cpu().numpy()  # 6M
                    entry_pred = torch.sigmoid(out["entry"]).cpu().numpy()
                    dd_pred = out["fwd_dd"].cpu().numpy()
                    for b_idx, seq in enumerate(blist):
                        for seq_pos in range(seq["valid_len"]):
                            key = (seq["ticker"], int(seq["fiscal_years"][seq_pos]))
                            pred_map[key] = {
                                "fwd_ret_6m": float(fwd_pred[b_idx, seq_pos]),
                                "entry": float(entry_pred[b_idx, seq_pos]),
                                "dd": float(dd_pred[b_idx, seq_pos]),
                            }
            # Map back to original npz rows
            for i, (t, fy_str) in enumerate(zip(tickers, fiscal_years, strict=False)):
                try:
                    fy = int(str(fy_str)[:4])
                    key = (t, fy)
                    if key in pred_map:
                        fwd_ret_pred[i] = pred_map[key]["fwd_ret_6m"]
                        entry_score[i] = pred_map[key]["entry"]
                        fwd_dd_pred[i] = pred_map[key]["dd"]
                except:
                    pass
            print(f"Loaded model predictions for {len(pred_map)} points")
        except Exception as e:
            print(f"Model prediction failed: {e}")
            import traceback

            traceback.print_exc()

    # Fallback heuristic if no model predictions: use true forward where available for backtest, and proxy entry score from price vs 52w + CEO change
    # For latest FY per ticker, compute score
    # Group by ticker -> latest FY row
    latest_per_ticker = {}
    for i, (t, fy_str) in enumerate(zip(tickers, fiscal_years, strict=False)):
        try:
            fy = int(str(fy_str)[:4])
        except:
            continue
        if t not in latest_per_ticker or fy > latest_per_ticker[t][0]:
            latest_per_ticker[t] = (fy, i)

    trades = []
    for t, (fy, idx) in latest_per_ticker.items():
        # Use pred if available else true for backtest demo
        ret_6m = (
            fwd_ret_pred[idx]
            if np.isfinite(fwd_ret_pred[idx])
            else fwd_ret_6m_true[idx]
        )
        dd_6m = (
            fwd_dd_pred[idx] if np.isfinite(fwd_dd_pred[idx]) else fwd_dd_6m_true[idx]
        )
        entry = (
            entry_score[idx]
            if entry_score[idx] != 0
            else (0.6 if triple_true[idx] == 1 else 0.4)
        )  # proxy

        # Heuristic: if we still have nan, estimate from features
        if not np.isfinite(ret_6m):
            # try to use last price momentum? skip
            continue

        # Apply filters
        if (
            entry >= entry_threshold
            and ret_6m >= fwd_ret_threshold
            and (np.isnan(dd_6m) or dd_6m >= dd_threshold)
        ):
            trades.append(
                {
                    "ticker": t,
                    "fiscal_year": fy,
                    "sector": str(sectors[idx]),
                    "price": float(price[idx]) if idx < len(price) else 0,
                    "entry_score": float(entry),
                    "fwd_ret_6m_pred": float(ret_6m),
                    "fwd_dd_6m_pred": float(dd_6m) if np.isfinite(dd_6m) else None,
                    "triple_true": int(triple_true[idx])
                    if triple_true[idx] != -1
                    else None,
                    "filter_pass": True,
                }
            )

    # Sort by entry_score * fwd_ret
    trades_sorted = sorted(
        trades, key=lambda x: x["entry_score"] * x["fwd_ret_6m_pred"], reverse=True
    )
    top = trades_sorted[:top_k]

    out_path = DATA_DIR / "trades_career.json"
    out_path.write_text(json.dumps(top, indent=2))
    print(f"Saved {len(top)} trades to {out_path} (total passing {len(trades_sorted)})")

    # Markdown table
    md_lines = [
        "# Career MTNN Trades",
        f"Filter: entry>{entry_threshold} & fwd_ret_6M>{fwd_ret_threshold} & dd>{dd_threshold}",
        f"Total passing: {len(trades_sorted)} / {len(latest_per_ticker)} tickers",
        "",
        "| Rank | Ticker | FY | Sector | Price | Entry | Fwd6M | DD6M | TripleTruth |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, tr in enumerate(top, 1):
        md_lines.append(
            f"| {i} | {tr['ticker']} | {tr['fiscal_year']} | {tr['sector']} | {tr['price']:.2f} | {tr['entry_score']:.3f} | {tr['fwd_ret_6m_pred']:.3f} | {tr['fwd_dd_6m_pred']} | {tr['triple_true']} |"
        )
    md_path = DATA_DIR / "trades_career.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote markdown {md_path}")
    return top


if __name__ == "__main__":
    score_trades()
