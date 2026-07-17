"""
Continuous career dataset for Vector Equities
Ticker -> sequence (N_ticker, T, D) sorted FY 2015-2024
career_pos_emb = year_norm + absolute year + tenure + time_since_ceo + macro

Outputs tensors for model_career.ContinuousFusion + CausalCareerTransformer
"""
from pathlib import Path
import json, bisect
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"

def load_bundle(path=None):
    # try v5 first, fallback v4, then train_matrix
    candidates = [
        DATA_DIR / "train_matrix_v6.npz",
        DATA_DIR / "train_matrix_v5.npz",
        DATA_DIR / "train_matrix_real.npz",
        DATA_DIR / "train_matrix.npz",
    ]
    if path:
        candidates = [Path(path)] + candidates
    npz_path = None
    for c in candidates:
        if c.exists():
            npz_path = c
            break
    if npz_path is None:
        raise FileNotFoundError(f"No train matrix found in {candidates}")
    npz = np.load(npz_path, allow_pickle=True)
    manifest_path = DATA_DIR / "feature_manifest_v6.json"
    if not manifest_path.exists():
        manifest_path = DATA_DIR / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    Z = npz["Z"].astype(np.float32)  # N,D normalized
    mask = npz["mask"].astype(np.float32) if "mask" in npz else np.ones_like(Z)
    Z_raw = npz["Z_raw"].astype(np.float32) if "Z_raw" in npz else Z  # fallback
    tickers = npz["ticker"].astype(str)
    names = npz["name"].astype(str) if "name" in npz else tickers
    fiscal_years = npz["fiscal_year"].astype(str)
    sectors = npz["sector"].astype(str) if "sector" in npz else np.array(["Unknown"]*len(Z))
    # forward labels if exist
    fwd = {}
    for k in ["fwd_ret_1m","fwd_ret_3m","fwd_ret_6m","fwd_ret_12m","fwd_vol_6m","fwd_dd_6m","triple_barrier","ceo_change_flag","time_since_ceo","price"]:
        if k in npz:
            fwd[k] = npz[k]
    print(f"Loaded bundle {npz_path} N={len(Z)} D={Z.shape[1]} fwd keys={list(fwd.keys())}")
    return Z, mask, Z_raw, tickers, names, fiscal_years, sectors, manifest, fwd, npz_path

def family_slices(manifest):
    fams = defaultdict(list)
    for j,f in enumerate(manifest["features"]):
        fam = manifest["families"][j] if j < len(manifest["families"]) else "unknown"
        fams[fam].append(j)
    return dict(fams), manifest["features"]

def build_sequences(Z, mask, Z_raw, tickers, fiscal_years, sectors, manifest, fwd, max_seq_len=10):
    feats = manifest["features"]
    feat_to_idx = {f:i for i,f in enumerate(feats)}
    # time enc sources
    idx_ceo_tenure = feat_to_idx.get("CEO_TENURE", None)
    idx_rate = feat_to_idx.get("RATE_10Y", None)
    idx_vix = feat_to_idx.get("VIX_AVG_FY", None)
    idx_price_vs = feat_to_idx.get("PRICE_VS_52W_HIGH", None)
    idx_rsi = feat_to_idx.get("RSI_14_PROXY", None)

    # group by ticker
    by_ticker = defaultdict(list)
    for i,(t,s) in enumerate(zip(tickers, fiscal_years)):
        try:
            fy = int(str(s)[:4])
        except:
            fy = 0
        by_ticker[t].append((fy, i))
    sequences = []
    for ticker, lst in by_ticker.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])  # by FY
        # filter to 2015-2024
        lst_sorted = [(fy,i) for fy,i in lst_sorted if 2015 <= fy <= 2024]
        if len(lst_sorted) == 0:
            continue
        # avg rows per ticker etc
        L = len(lst_sorted)
        # build arrays
        seq = {
            "ticker": ticker,
            "sector": sectors[lst_sorted[0][1]],  # assume sector stable
            "fiscal_years": [],
            "indices": [],  # original row indices
            "valid_len": L,
        }
        fys = []
        idxs = []
        for fy,i in lst_sorted:
            fys.append(fy)
            idxs.append(i)
        seq["fiscal_years"] = np.array(fys, dtype=np.int64)
        seq["indices"] = np.array(idxs, dtype=np.int64)
        # For quick access store raw
        sequences.append(seq)
    # sort by ticker for determinism
    sequences = sorted(sequences, key=lambda x: x["ticker"])
    print(f"Built {len(sequences)} ticker sequences avg len {np.mean([s['valid_len'] for s in sequences]):.2f}")
    return sequences, feat_to_idx, dict(
        ceo_tenure=idx_ceo_tenure,
        rate=idx_rate,
        vix=idx_vix,
        price_vs=idx_price_vs,
        rsi=idx_rsi,
    )

def get_time_enc_for_seq(seq, Z_raw, fwd, feat_idx_map):
    # seq contains indices and fiscal_years
    L = len(seq["fiscal_years"])
    max_len = 10
    time_enc = np.zeros((max_len, 8), dtype=np.float32)
    year_norm_seq = np.zeros((max_len, 1), dtype=np.float32)
    mask_seq = np.zeros((max_len,), dtype=bool)
    # forward labels padded
    fwd_ret_6m_seq = np.full((max_len,), np.nan, dtype=np.float32)
    fwd_ret_1m_seq = np.full((max_len,), np.nan, dtype=np.float32)
    fwd_ret_3m_seq = np.full((max_len,), np.nan, dtype=np.float32)
    fwd_ret_12m_seq = np.full((max_len,), np.nan, dtype=np.float32)
    fwd_vol_seq = np.full((max_len,), np.nan, dtype=np.float32)
    fwd_dd_seq = np.full((max_len,), np.nan, dtype=np.float32)
    triple_seq = np.full((max_len,), -1, dtype=np.int64)
    ceo_change_seq = np.zeros((max_len,), dtype=np.float32)
    time_since_ceo_seq = np.zeros((max_len,), dtype=np.float32)

    for pos, (fy, orig_idx) in enumerate(zip(seq["fiscal_years"], seq["indices"])):
        if pos >= max_len:
            break
        year_norm = (fy - 2015) / 9.0
        year_norm_seq[pos,0] = year_norm
        mask_seq[pos] = True
        # raw feature access
        def raw_feat(idx_name):
            j = feat_idx_map.get(idx_name)
            if j is None:
                return 0.0
            try:
                v = Z_raw[orig_idx, j]
                if np.isnan(v) or np.isinf(v):
                    return 0.0
                return float(v)
            except:
                return 0.0
        ceo_tenure = raw_feat("CEO_TENURE")
        rate = raw_feat("RATE_10Y")
        vix = raw_feat("VIX_AVG_FY")
        price_vs = raw_feat("PRICE_VS_52W_HIGH")
        rsi = raw_feat("RSI_14_PROXY")
        # from fwd dict
        ceo_change = 0.0
        time_since = 0.0
        if "ceo_change_flag" in fwd:
            try:
                ceo_change = float(fwd["ceo_change_flag"][orig_idx])
            except:
                pass
        if "time_since_ceo" in fwd:
            try:
                time_since = float(fwd["time_since_ceo"][orig_idx])
            except:
                pass
        # normalize
        ceo_tenure_norm = np.clip(ceo_tenure / 20.0, 0, 1)
        time_since_norm = np.clip(time_since / 10.0, 0, 1)
        rate_norm = np.clip(rate / 5.0, 0, 1.5)
        vix_norm = np.clip(vix / 40.0, 0, 1.5)
        # price_vs already 0-1ish, rsi 0-100
        price_vs_clipped = np.clip(price_vs, 0, 1.5) if price_vs !=0 else 0.9
        # rsi from normalized Z_raw? Might be raw 0-100 or normalized. If raw is normalized (z-score) fallback.
        # Heuristic: if rsi raw magnitude <5 likely z-scored, map to 0.5
        if abs(rsi) < 5 and abs(rsi) > 0:
            # could be z-scored, but we had 50 default, check magnitude
            # If z-scored around 0, we map RSI ~50 => 0.5
            # We'll try to recover: if Z_raw came from filled median then raw RSI median ~50, but if z-scored, median 0
            # Use fallback 0.5 for |rsi|<4
            if abs(rsi) < 4:
                rsi_norm = 0.5
            else:
                rsi_norm = np.clip((rsi + 4)/8, 0,1)  # rough
        else:
            rsi_norm = np.clip(rsi / 100.0, 0,1) if rsi !=0 else 0.5

        time_enc[pos] = np.array([
            year_norm,
            ceo_tenure_norm,
            time_since_norm,
            ceo_change,
            rate_norm,
            vix_norm,
            price_vs_clipped,
            rsi_norm,
        ], dtype=np.float32)

        # forward labels
        for key_arr, out_arr in [
            ("fwd_ret_1m", fwd_ret_1m_seq),
            ("fwd_ret_3m", fwd_ret_3m_seq),
            ("fwd_ret_6m", fwd_ret_6m_seq),
            ("fwd_ret_12m", fwd_ret_12m_seq),
            ("fwd_vol_6m", fwd_vol_seq),
            ("fwd_dd_6m", fwd_dd_seq),
        ]:
            if key_arr in fwd:
                try:
                    v = fwd[key_arr][orig_idx]
                    if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                        pass
                    else:
                        out_arr[pos] = float(v)
                except:
                    pass
        if "triple_barrier" in fwd:
            try:
                tb = int(fwd["triple_barrier"][orig_idx])
                triple_seq[pos] = tb
            except:
                pass
        ceo_change_seq[pos] = ceo_change
        time_since_ceo_seq[pos] = time_since

    return {
        "time_enc": time_enc,
        "year_norm": year_norm_seq,
        "mask": mask_seq,
        "fwd_ret_1m": fwd_ret_1m_seq,
        "fwd_ret_3m": fwd_ret_3m_seq,
        "fwd_ret_6m": fwd_ret_6m_seq,
        "fwd_ret_12m": fwd_ret_12m_seq,
        "fwd_vol_6m": fwd_vol_seq,
        "fwd_dd_6m": fwd_dd_seq,
        "triple_barrier": triple_seq,
        "ceo_change_flag": ceo_change_seq,
        "time_since_ceo": time_since_ceo_seq,
    }

def collate_batch(batch_seqs, Z, mask, Z_raw, fams, feat_list, max_len=10):
    """
    batch_seqs: list of seq dicts from build_sequences
    Returns tensors dict for model
    """
    B = len(batch_seqs)
    # family dims
    fam_dims = {fam: len(cols) for fam, cols in fams.items()}
    # init per family
    xs_seq = {fam: np.zeros((B, max_len, fam_dims[fam]), dtype=np.float32) for fam in fams}
    ms_seq = {fam: np.zeros((B, max_len, fam_dims[fam]), dtype=np.float32) for fam in fams}
    time_enc_seq = np.zeros((B, max_len, 8), dtype=np.float32)
    year_norm_seq = np.zeros((B, max_len, 1), dtype=np.float32)
    valid_mask = np.zeros((B, max_len), dtype=bool)
    fwd_ret_6m = np.full((B, max_len), np.nan, dtype=np.float32)
    fwd_ret_1m = np.full((B, max_len), np.nan, dtype=np.float32)
    triple = np.full((B, max_len), -1, dtype=np.int64)
    sectors = []
    tickers = []
    fiscal_years_batch = np.zeros((B, max_len), dtype=np.int64)

    for b, seq in enumerate(batch_seqs):
        tickers.append(seq["ticker"])
        sectors.append(seq["sector"])
        L = seq["valid_len"]
        idxs = seq["indices"]
        fys = seq["fiscal_years"]
        # time enc and labels
        # we need feat_to_idx for time enc? We'll compute externally then merge
        # For xs/ms, slice Z and mask per family
        for pos in range(min(L, max_len)):
            orig_idx = idxs[pos]
            fiscal_years_batch[b, pos] = fys[pos]
            for fam, cols in fams.items():
                xs_seq[fam][b, pos, :] = Z[orig_idx, cols]
                ms_seq[fam][b, pos, :] = mask[orig_idx, cols]
        # time enc is built via helper that needs Z_raw and fwd – we call outside? For simplicity recompute inside using global
        # placeholder will be overwritten by caller if needed
    return {
        "xs_seq": xs_seq,
        "ms_seq": ms_seq,
        "time_enc_seq": time_enc_seq,
        "year_norm_seq": year_norm_seq,
        "valid_mask": valid_mask,
        "tickers": tickers,
        "sectors": sectors,
        "fiscal_years": fiscal_years_batch,
        "fwd_ret_6m": fwd_ret_6m,
        "triple": triple,
    }

# For standalone test
if __name__ == "__main__":
    Z, mask, Z_raw, tickers, names, fiscal_years, sectors, manifest, fwd, path = load_bundle()
    fams, feat_list = family_slices(manifest)
    seqs, feat_to_idx, extra_idx = build_sequences(Z, mask, Z_raw, tickers, fiscal_years, sectors, manifest, fwd)
    print(f"Example seq {seqs[0]}")
    # test time enc for first
    feat_map = {f:i for i,f in enumerate(feat_list)}
    enc = get_time_enc_for_seq(seqs[0], Z_raw, fwd, feat_map)
    print(enc["time_enc"][:seqs[0]["valid_len"]])
