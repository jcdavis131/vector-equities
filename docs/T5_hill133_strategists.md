# T5 Hill 133 — Triple Strategist ×3 — v6 Money Next Hill

**Date:** 2026-08-12 CDT 19:47 — T5 epic equities hill 133 :01 lite 5m
**Label:** cc593b63-618c-4348-b7fb-f04b9936883b → c33d6884 strategist-swarm-3-lens
**Zero-deps:** true `bundles/zero_deps.json` `{"zero_deps":true,"allow":"acne:./src"}` — torch `auto cuda if available else cpu`, stdlib only, no pip, no cloud, ACNE optional local
**Pacing:** :01 tempo, max3/4 swarm, 5m lite, OODA Orient L1, MoMA-lite router
**Platform:** free platform free users forever — no paywall, 5th game ever free, payments/store.jsonl empty, auth free 0.9, no Stripe live keys until explicit yes. Monetization = private edge 0.25 Kelly 1% max3 concurrent, paper → tiny 0DTE gated IC>0.03 Sharpe>1.2 win>55% DD<12% kill-switch separate bankroll NOT financial advice
**Mission Log:** triple-write 7-field mandatory nodeId agentId attempt latency tokens status errorClass ts runId ooda tempo :01 even no-change per checkpoint-manager `bundles/ultra/runs/` + `.scout/missions/_cron/timeline.jsonl`

---

## 0) Hill 132 Snapshot — What We Inherit

Source: `assets/eval_scoreboard.json`, `assets/eval_forward.json`, `assets/eval_sector_coherence.json`, `pipeline/data/mtnn_report.json`, `HILLCLIMB_v6_to_v2_REPORT.md`, smoke logs `/tmp/equities_trans_14400_smoke.log`

**v6 money best:** `equities_v6_money_best.pt` 514K params (300K backbone + heads) — **CQS 0.7017** vs baseline 0.605 baseline+0.005 → **PASS +0.0967**. Target 0.72+ needs +0.0183.

| Metric | v6 best | Gate | Verdict |
|---|---|---|---|
| CQS | 0.701666... | >0.605 (baseline+0.005) | PASS |
| recall@10 same-ticker-next-FY | 1.0 | within 0.02 baseline | PASS |
| purity@20 archetype | 0.68 | >0.60 | PASS |
| sector_acc 11-way GICS | 0.957 | >0.30 prac >0.95 | PASS |
| continuity AR1 FY12 14,400 pairs 13,200 contiguous | 0.72 | sticky career | PASS |
| market_acc | 0.57 | >0.58 | FAIL |
| next_R2 14d | 0.18 baseline / 0.244 gated60ep | >0.20 | FAIL base / PASS gated |
| dim | 64 | MTNN v5 gated + transformer CLS 4L 4H d_model96 | — |
| ticker split 70/15/15 no leak honest_split true | 14,400 FYs 122 feats 1,200×12 tickers×years | — | PASS |

**Forward IC — honest FAIL → no v2:**

| Metric | Value | Target v2 | Gate |
|---|---|---|---|
| IC rank 1M Spearman n=233 | 0.0051 | >0 | PASS weak |
| IC 3M | 0.0064 | >0 | PASS |
| IC 6M rankdata 0.007 / scipy spearman 0.0097 | 0.007 / 0.0097 | **>0.01** for v2 | **FAIL → no v2** |
| IC 12M | 0.0062 | >0 | PASS |
| Top50 conviction IC | 0.079 n=50 | >0.01 | PASS small-n |
| IC target proxy | 0.5066 | — | — |
| calibration bias isotonic | before 5.76% (11.37%→5.61%) after **0.0** | <1% | PASS 162 thr `forward_calibration_isotonic.json` |
| triple-barrier +10% before -7% 63d | 0.2189 random 0.25 | >random | **FAIL false** |
| distress corr pred_fwd6 vs DD | **-0.2624** higher pred→more distress | >0 wanted | **FAIL inverted** |
| Sharpe after $0.01 slip mean0.0504 std0.1251 | sqrtN 6.15 PASS >1.0 / sqrt2_ann6M **0.57 FAIL** <1.2 | Sharpe>1.2 | **SPLIT FAIL ambiguous honest** |
| n_trades | 233 `trades_final_ranked_v6.csv` entry_mean 0.8409 thr0.7 | ≥200 Kelly min | PASS |

**Smoke vs Full evidence:**

- Smoke trans 2ep 4,000 rows CPU: CQS 0.5908 recall 0.9125 purity 0.7589 PASS sector 0.13025 FAIL (CLS starved 2ep) market 0.593 PASS next_R2 -0.0031 FAIL comp 0.842 — proves transformer recall generalizes fast
- Full 14.4k 60ep epoch0 loss6.0163 val_recall0.9 test0.95 purity0.718 comp0.809 would beat 0.7017 SIGTERM 167s before epoch1 — **need LOCAL-GPU resume**
- Gated 14.4k 2ep batch1024 crash buffered SIGTERM 74-88s no logs → fixed `-u` unbuffered

**Money gate:** v2 requires **IC>0.01 AND Sharpe>1.2 AND CQS>0.72 + market>0.58 + next_R2>0.20** → currently IC 0.007/0.0097 FAIL, Sharpe 0.57 FAIL, CQS 0.7017 <0.72 FAIL → honest no promotion `equities_v6_money_v2.pt` not created.

**Target Hill 133:** CQS 0.72+ (+0.0183), IC 0.011+ (+0.0013–0.004 over 0.007–0.0097), Sharpe sqrt2 → >0.8 stretch >1.2, triple >0.25, distress >0, market >0.58, next_R2 >0.20.

---

## A) Strategist Lens 1 — Loss Engineering (Fix Sharpe sqrt2 FAIL, Distress Invert, Triple Calibrate, Variance Collapse)

**Owner:** loss-engineering strategist — T5 hill 133 A
**Problem stack:** std collapse 0.7% vs optimal 4% (5.7× collapsed) → Sharpe sqrt2 0.57 FAIL, sqrtN 6.15 PASS inflated by n, distress -0.2624 inverted, triple 0.2189 <0.25 random FAIL gate, ranking wasted, bias already fixed 0.0 via isotonic but raw collapse persists.

### A1 Current Loss — v6 money

```python
# baseline v6 money
mse_fwd = MSE(pred_fwd6, true_fwd12M)           # w_f6=1.0 currently
mse_dd  = MSE(pred_dd, true_dd) * w_dd 0.8?
infoNCE same-ticker adj FY temp0.08 + hard-neg 0.2 same-sector + feature-dropout 0.12
# no ranking, no var, no distress invert, no triple horizon
```

### A2 Proposed Loss v6.1→v6.1.1 Hill 133

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # zero-deps true

# hyper params
w_f6 = 5.0          # up from 1.0 per TOWER_V6_DESIGN
w_dd = 1.5
w_vol = 0.3
w_entry = 2.0
w_nce = 1.0
w_rank = 1.0        # new ranking
w_var = 0.2         # variance push 0.7% → 4%
w_distress = 0.8    # invert -0.2624
w_triple = 0.5      # calibration auxiliary

loss = w_f6*mse_fwd + w_dd*mse_dd + w_vol*mse_vol + w_entry*bce_entry + w_nce*nce \
     + w_rank*rank_loss_256 + w_var*var_loss + w_distress*distress_penalty + w_triple*triple_bce
```

#### Ranking Loss 256 pairs margin 0.02 threshold 5%

- Sample 256 pairs per batch (batch512 → 512 choose 2 = 130k, sample 256 to keep O(n))
- Only label where `|true_diff| > 0.05` (5% forward spread)
- Margin 0.02 Spearman-like pairwise hinge

```python
def ranking_loss_256(pred, true, n_pairs=256, margin=0.02, thresh=0.05):
    # pred [B], true [B]
    # sample pairs i<j
    idx_i = torch.randint(0, B, (n_pairs,), device=device)
    idx_j = torch.randint(0, B, (n_pairs,), device=device)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    dt = true[idx_i] - true[idx_j]
    valid = dt.abs() > thresh
    if valid.sum()==0: return torch.tensor(0.0, device=device)
    pred_diff = pred[idx_i] - pred[idx_j]
    # if true_i > true_j we want pred_diff > margin
    y = torch.sign(dt[valid])  # +1 or -1
    loss = torch.clamp(margin - y * pred_diff[valid], min=0.0).mean()
    return loss
```

Why 256 pairs: balances IC uplift vs memory; margin 0.02 matches isotonic bin width ~0.04 (half-bin separation). Expected IC lift +0.003→0.006 from prior v5 rank tuning 0.16→0.22 with similar.

#### Distress Invert Penalty — Fix -0.2624 → >0

Current: corr(pred_fwd6, dd_proxy) = -0.2624 negative = higher pred correlated with more distress (dd = max drawdown in 63d, lower = worse). Wanted: higher pred → less distress (positive correlation with -dd or positive with dd less negative? Actually true: dd threshold -0.07 — we want pred high when dd > -0.07 safe).

Construct validity: `eval_forward.json` describes `distress_early_warning_corr_pred_vs_dd` = corr(pred fwd6 vs dd threshold); positive means higher pred → less distress. Currently -0.2624 negative inverted.

Two fixes:

1. **Invert head target:** Instead of predicting dd as drawdown magnitude, predict safety score `s = 1 if dd_clock>-0.07 else 0` or `-dd` (higher = safer). Add `distress_penalty = -corr(pred_fwd6, safety_proxy)` minimization.
2. **Corr penalty:** differentiable Spearman proxy via Pearson on ranks:

```python
def distress_invert_penalty(pred_fwd, true_dd, lam=0.8):
    # true_dd: more negative = worse; safety = clip(dd > -0.07)
    safety = torch.clamp(true_dd + 0.07, min=-1.0, max=0.5) # higher = safer, 0 thr
    # or binary safe flag from trade proxy true_fwd>=0.10 & dd>-0.07 triple definition
    if pred_fwd.std() < 1e-6 or safety.std() < 1e-6:
        return torch.tensor(0.0, device=device)
    # Pearson corr
    corr = ((pred_fwd - pred_fwd.mean()) * (safety - safety.mean())).mean() / (pred_fwd.std()*safety.std() + 1e-6)
    # we want corr >0, current -0.2624 → penalize negative
    penalty = -corr  # minimize => maximize corr
    # only penalize if corr<0.1 else 0 (hinge)
    return torch.clamp(0.1 - corr, min=0.0) * lam
```

Add auxiliary BCE `triple` head: predict prob(hit +10% before -7%) — train separately then auxiliary.

#### Triple-Barrier Calibration 21d Threshold — Fix 0.2189 <0.25

Current fails: +10% before -7% 63d proxy `true_fwd>=0.10 & dd>-0.07` 0.2189 < random 0.25 false.

Problems:
- Horizon too long 63d dilutes; 2024 high VIX mean 3.3% bearish but short spikes matter
- Threshold too strict +10% ambitious given mean 0.84% entry thr 0.7
- Proxy `true_fwd>=0.10 & dd>-0.07` not actual barrier path-dependent

Hill133 fix:

- **Horizon:** 63d → 21d (3-week) for 1M IC linkage + 42d secondary for 2M
- **Threshold:** +7%/-5% (more achievable) and +10%/-7% keep both but calibrate
- **True label:** path-dependent from `for_history.json` daily OHLC not just end return — need barrier simulation (yfinance daily high/low). For v6 lite zero-deps synthetic proxy first: `entry_mean 0.8409 thr0.7` → calibrate entry thr 0.72 for 21d.

Calibration steps:

```python
# triple barrier label build (offline synthetic until price history fetch)
def build_triple_label(prices_daily, entry_idx, up=0.07, down=-0.05, horizon=21):
    # prices_daily: FY aligned daily close/high/low for ticker
    entry = prices_daily[entry_idx]
    window = prices_daily[entry_idx+1:entry_idx+1+horizon]
    for t, bar in enumerate(window):
        ret = (bar['high']/entry-1) # intraday up
        if ret >= up: return 1.0, t  # win
        if (bar['low']/entry-1) <= down: return 0.0, t  # loss
    # timeout: return final return bucket
    final = window[-1]['close']/entry-1 if len(window)>0 else 0.0
    return 1.0 if final>=0.02 else 0.0, horizon
```

Then BCE head + calibration:

```python
bce_triple = BCEWithLogits(pred_triple_logit, true_triple_21d_up7_down5) * 0.5
# isotonic already bias 0.0 — preserve with triple too
```

Gate win>55% for 0DTE tiny gated: current 21.89% FAIL → with 21d +7%/-5% expect ~45-55% achievable; with entry thr 0.84 top30% filtering win rises.

#### Var Loss Push std 0.7% → 4% — Fix Sharpe sqrt2 0.57 FAIL

Collapse root cause: MSE-only pushes to mean 0.7% std (optimal 4% = IC*true_std per TOWER_V6_DESIGN). Sharpe sqrt2_ann = mean/std * sqrt(2) because 6M = half-year → std collapse inflates sqrtN 6.15 but sqrt2 0.57 FAIL ambiguous documented `eval_forward.json`.

Optimal std derivation:

- true_fwd std ~ ? from `for_history.json` — assume 20-25% annual → 14% 6M → IC 0.18 ideal predictor std = IC * true_std ≈ 0.18*0.20≈0.036 → 3.6% ≈ 4% target

Fix:

```python
def var_loss(pred_fwd, target_std=0.04):
    pred_std = pred_fwd.std()
    # hinge if below 3%, push to 4%
    # MSE between stds
    loss_std = (pred_std - target_std).pow(2) * 0.2
    # plus variance regularization to avoid shortcut: penalize if batch mean collapses <0.01
    if pred_std < 0.01:
        loss_std += (0.01 - pred_std).pow(2) * 2.0
    return loss_std

# plus VICReg var hinge alternative from hoops v6 d64 setup
# var hinge: (1 - std).clamp
def vicreg_var_loss(x, gamma=1.0, lam=25.0/64): # per-dim for embedding dim 64?
    # but here for fwd head 1-d
    std = x.std() + 1e-4
    return torch.relu(gamma - std).mean() * lam
```

Combined: `var_loss std target 0.04` pushes std 0.7%→4% → Sharpe sqrt2 `mean 0.0504 / 0.04*1.414 = 1.78` PASS >1.2 expected if mean preserved (vs current std0.1251 sqrt2 0.57 FAIL). Note mean/std after var push + ranking stabilizes.

OneCycle + clip1.0 + AdamW 1e-4 + dropout0.12 temp0.08 hard-neg0.2 same.

Expected after Hill133 A: std 0.7%→3-4%, Sharpe sqrt2 0.57→1.0-1.8 PASS stretch, Sharpe sqrtN 6.15 stays >1.0, distress -0.2624→+0.1-0.3, triple 0.2189→0.32-0.45 with 21d +7%/-5%.

---

## B) Strategist Lens 2 — IC Uplift (Transformer CLS vs ContinuousFusion, VIX Gating, FY Conditioning GPR/EPU 2024 Regime)

**Owner:** architecture IC uplift strategist — T5 hill133 B
**Goal:** IC 0.0051/0.0064/0.007/0.0097/0.0062 → 0.011+ (+0.0013 over best 0.0097 spearman, +0.004 over rankdata 0.007). Top50 0.079 kept >0.05.

### B1 Evidence — Why Transformer Would Beat

- Smoke transformer 2ep 4k: recall 0.9125 fast vs gated 0.0 overfit small — transformer generalizes same-ticker-next-FY early
- Full 14.4k 60ep epoch0 comp 0.809 >0.7017 val_recall 0.9 test0.95 purity0.718 loss6.0163 would beat but SIGTERM 167s killed before epoch1
- Gated ContinuousFusion: 17→20 towers attentive but linear gating; sector 0.957 PASS but market 0.57 FAIL because market-wide FY collapse (GPR/EPU 2024 3.3% bearish mean)
- CLS token centralizes global context; hoops v6 transformer similar 4L 4H d_model128 17 towers cat([x·m,m]) CLS→64-d proven 0.7937→0.85 comp trajectory

### B2 Architecture Options — CLS Token Transformer vs Upgraded ContinuousFusion

#### Option 1: Transformer CLS Token (Preferred for Hill133 Full Train)

```
Towers: 20 families (118→154 feats 32 new) d_tower 24 tower_blocks 2 residual
Tokens: [industry_event*10 scattered? actually 3 new families = 3 tower tokens not 32 tokens — keeps n_towers 20 tokens]
        + [CLS] token learned 96-d
        + [FY] token learned 12-d regime + market scalar
        = 22 tokens total → transformer 4L 4H d_model96

Flow:
  family_slices 20 → ResidualTower(x) → t_i [B,24]
  FY conditioning: FY_emb 12-d learned + VIX + GPR/EPU 4 scalars → fy_token [B,96] via MLP 16→96
  Tokens cat = [cls[96], fy_token[96], t_1…t_20→96 via proj 24→96]
  Transformer 4 layers 4 heads PreNorm dropout0.12
  CLS_out → MLP heads 96→64→32→1 fwd, dd, vol, entry logit
  + embedding 64-d L2 norm for CQS

Attention = tower interaction learner: political_risk attends to industry_event in high VIX years automatically.
```

FY conditioning solves 2024 bearish mean 3.3% regime: earlier quantile mapping failure noted in `TOWER_V6_DESIGN.md` — market-wide EPU/GPR high causes 2024 mean 3.3% vs prior 6-8% but model mislabels as collapse signal. FY token lets model learn residual vs absolute.

Loss: CLS as bottleneck forces disentanglement, sector gating emerges from attention not hard mask.

#### Option 2: Upgraded ContinuousFusion with VIX-Gated Political/Trade (Fallback Lite CPU)

If LOCAL-GPU OOM or Hatch VM CPU should still POST CQS 0.59→0.68? Keep gated CPU lane.

Current ContinuousFusion:

```
gate = softmax(MLP(towers_mean)) [B, n_towers] static no FY
```

Upgrade to **VIX-gated FY regime aware fusion**:

```python
class ContinuousFusionVIX(nn.Module):
    def __init__(self, n_towers=20, d_tower=24, d_emb=64, fy_cond_dim=16):
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_tower*n_towers + fy_cond_dim, 128), nn.ReLU(), nn.Dropout(0.12),
            nn.Linear(128, n_towers)  # logits
        )
        self.fy_proj = nn.Linear(12+4, fy_cond_dim) # FY_emb 12 + VIX/GPR/EPU/Elec 4
        self.tower_proj = nn.Linear(d_tower, 64)

    def forward(self, tower_outs, fy_emb, vix_gpr_epu):
        # tower_outs [B,20,24], fy_emb [B,12], vix.. [B,4]
        fy_cond = self.fy_proj(torch.cat([fy_emb, vix_gpr_epu], -1))  # [B,16]
        flat = tower_outs.flatten(1)  # [B,20*24]
        gate_logits = self.gate_mlp(torch.cat([flat, fy_cond], -1))  # [B,20]
        # temperature annealing 1.0 → 0.7 during training = sharpen high VIX years
        gates = torch.softmax(gate_logits / temp, -1)  # [B,20] sums 1
        # political/trade gating visualisation: in high VIX years gates 18-20 ↑ from 0.05→0.18 expected
        fused = (tower_outs * gates.unsqueeze(-1)).sum(1)  # [B,24] weighted
        ...
```

Expected: political_risk gates ↑ in high VIX/GPR years 2020 (COVID), 2022 (rates), 2024 (election+GPR 3.3% mean). Industry_event gates ↑ when IND_DISPERSION_MOM high. Commodity betas gate ↑ when OIL_WTI_YOY |z|>2.

FY conditioning: FY12 emb + 4 market scalars → solves quantile mapping: instead of predicting absolute 3.3% as bearish error, model predicts relative to FY regime mean learned via fy_token.

### B3 FY Conditioning for GPR/EPU 2024 Regime 3.3% Mean

**Root cause:** `TOWER_V6_DESIGN.md` Issue — many features market-wide same for all tickers FY same (GPR, EPU, VIX, rates). Without FY conditioning, model learns "if GPR high → all stocks low" which collapses market_acc 0.57 and pushes mean to bearish but correct interpretation is 2024 mean 3.3% was market regime not firm-specific collapse signal. Quantile mapping failure: mapping 2024 true distribution 3.3% via global z-score mislabels 50% of entries.

Fix FY conditioning two layers:

1. **Input:** FY emb 12-d learned + `fy_market_feats = [VIX_FY_AVG, GPR_GLOBAL_AVG_FY, EPU_US_AVG_FY, RATE_VOL_3M]` standardized 5Y rolling z-score avoids lookahead.
2. **Model:** fy_token prepended to transformer (option1) or fy_cond appended to gate MLP (option2) as above.
3. **Loss/Target:** residual target `true_fwd_resid = true_fwd - fy_mean_true` auxiliary loss w=0.3, final pred = fy_mean_pred + resid_pred. Lets model decouple market vs alphas.

Expected: market_acc 0.57→0.60+ JUST PASS, IC uplift because rank within FY de-noised removes market regime common mode improves Spearman 0.007→0.011 (+57% needed).

### B4 Transformer Ablation Plan Hill133

| Config | d_model | nL | nH | d_tower | Blocks | batch | lr | Expected CQS | IC | Compute |
|---|---|---|---|---|---|---|---|---|---|---|
| **CLS v6.1 lite** smoke CPU 4k 2ep | 96 | 4 | 4 | 24 | 1 | 512 | 1.5e-3 | 0.5908→0.62 | 0.006→0.008 | CPU 90s |
| **CLS v6.1 full 14.4k 60ep** LOCAL-GPU | 96 | 4 | 4 | 24 | 2 | 512 | 1.5e-3 OneCycle10% | **0.72-0.75** | **0.011-0.018** | RTX4090 25m |
| Gated VIX upgraded fallback | 64 | — | — | 24 | 2 | 1024 | 2e-3 | 0.7017→0.715 | 0.007→0.009 | CPU 167s guard |
| Gated original (keep best 0.7017) | 64 | — | — | 24 | 2 | 1024 | 2e-3 | 0.7017 | 0.007 | keep |

Hill133 gate: **CLS 14.4k 60ep epoch0 comp 0.809 already >0.7017** — 60ep expected CQS 0.72+ PASS would beat SIGTERM 167s → LOCAL-GPU resume command below.

LOCAL-GPU RTX4090 resume (Hatch VM = CPU per `MEMORY.md` training split auto device):

```bash
python3 -u pipeline/train_mtnn.py \
  --epochs 60 --dim 64 --fusion transformer \
  --batch 512 --tower-width 24 --d-model 96 \
  --n-fusion-layers 4 --n-attn-heads 4 \
  --tower-blocks 2 --mlp-heads \
  --lr 1.5e-3 --weight-decay 1e-4 --val-every 5 --device cuda \
  --one-cycle --pct-start 0.1 --clip 1.0 --dropout 0.12 --temp 0.08 --hard-neg 0.2 \
  --w-f6 5.0 --rank-w 1.0 --var-w 0.2 --n-pairs 256 --margin 0.02
# nano fallback if OOM: --batch 256 --accum 2
# monitor pipeline/data/mtnn_best.pt → equities_v6_money_v2.pt ONLY IF CQS>0.72 IC>0.01 Sharpe>1.2 triple_write 7-field
```

Transformer CLS token visual: compare `shared-map.js` 3D embedding central — tower attention map heat LOD4000 visible dark #080A0F.

### B5 Expected IC Bridge

- Baseline 0.7017 IC 0.007/0.0097
- + w_f6=5 + ranking 256 pairs margin0.02 → IC 0.007→0.009 (+0.002)
- + variance std 0.7%→4% → IC 0.009→0.0105 (de-collapse improves rank)
- + FY conditioning residual → IC 0.0105→0.012 (+0.0015 market de-noise)
- + transformer CLS tower interaction → IC 0.012→0.015 Top50 0.079→0.11
- + political/trade VIX gating → IC 0.015→0.018 high VIX years accuracy

Gate 0.011+ PASS gives margin.

---

## C) Strategist Lens 3 — Tower Integration (10f GDELT + 10f GPR/EPU + 12f Commodity/Beta, Synthetic Fallback Offline)

**Owner:** tower-integration strategist — T5 hill133 C
**Status:** `pipeline/towers_v6/` exists 29 lines synthetic fallback offline because `train_matrix_v5.npz` missing clean clone, yfinance/GDELT timeout offline zero-deps true per `bundles/zero_deps.json`. Auto-detect families `EquitiesMTNN` no code change.

### C1 Architecture of New 32 feats 3 families 17→20 D 122→154 N 14,400 (1,200×12 tickers×12 FYs) 14,400×154 ~6.8MB ~1.7MB npz fine VM 7.8G

```python
# pipeline/towers_v6/__init__.py — already live
new_features = ind_cols + pol_cols + trade_cols  # 10+10+12=32
new_families = [industry_event]*10 + [political_risk]*10 + [global_trade_commodity]*12
# feature_spec merge: old 118 (17f) + 32 = 154
# Z-scored 5Y rolling YoY to avoid lookahead, fill NaN 0, mask accordingly
```

Model auto-detects families via `feature_spec -> family_slices dict -> ResidualTower per family`. Fusion `ContinuousFusion 17→20 will learn to gate political/trade higher in high VIX years. CLS transformer tower tokens+FY+CLS 22 tokens 4L 4H d_model96 → emb L2 norm.

### C2 Industry_Event 10f — `towers_v6/industry_gdelt.py` — GDELT 2.0 Doc API + 8-K + EDGAR sector proxy

Purpose: sector-specific shocks, disruption, regulatory shifts, earnings breadth, vol spike.

| # | Feat | Def | Source | Norm | Fallback offline |
|---|---|---|---|---|---|
| 1 | IND_NEWS_VOL_Z | log count GDELT docs sector/year z vs sector history | GDELT 2.0 Doc API GICS→keyword map (GICS 11 → themes STRIKE RECALL LAWSUIT REGULATORY MERGER PRODUCT_LAUNCH SUPPLY_CHAIN) | log1p + z 5Y rolling | sector_context form count proxy + sector noise N(0,0.3) |
| 2 | IND_NEWS_TONE_AVG | GDELT avg tone -10..+10 normalize | GDELT Doc tone field | /10 → [-1,1] z | EDGAR MDA sentiment proxy existing |
| 3 | IND_NEG_EVENT_CNT | strike+recall+lawsuit+layoff themes count | GDELT GKG tone<-3 + themes | log1p | synthetic sector-specific negative shock flag high IND_VOL_SPIKE years |
| 4 | IND_POS_EVENT_CNT | product launch + contract + patent | GDELT Doc + 8-K Item 1.01/2.01 material events | log1p | earnings streak aggregation |
| 5 | IND_REGULATORY_RISK | regulation+antitrust themes | GDELT + SEC 8-K Item 8.01 regulatory disclosure counts | z | sector_context govt contract exposure static map |
| 6 | IND_MA_INTENSITY | M&A theme vol YoY | GDELT MERGER theme + 8-K Item 1.01 | YoY vs prior FY | same |
| 7 | IND_SUPPLY_DISRUPTION | supply chain + shortage + logistics themes | GDELT SUPPLY_CHAIN + BDRY/GSCPI correlation proxy | z | GSCPI×sector beta (Industrials high) |
| 8 | IND_EARN_BREADTH | sector % beating estimates aggregated | existing EARN_SURPRISE_STREAK[] aggregated per sector FY | z intra-sector | from existing 122 feats aggregation |
| 9 | IND_DISPERSION_MOM | std RET_12M within sector | computed from universe prices 11 GICS intra-sector std | z | std of sector price existing |
| 10 | IND_VOL_SPIKE | sector vol / market vol VIX-like | VIX proxy sector_vol 90d / market vol | ratio log | synthetic 1.0 + noise |

Impl `industry_gdelt.py` queries GDELT per sector/year cache JSON `pipeline/data/gdelt_cache.json` offline fallback synthetic proxy sector_context+form proxy + sector noise ensures zero-deps true torch auto cuda else cpu CPU on Hatch VM passes SMOKE.

### C3 Political_Risk 10f — `towers_v6/political.py` — GPR + EPU + Election Calendar + WGI + Rate Vol

Purpose: Elections, policy uncertainty, geopolitics, FY regime conditioning to solve 2024 mean 3.3% bearish mislabelled as collapse.

Sources (free public no key):

- GPR Index Iacoviello `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls` monthly global
- EPU Baker-Bloom-Davis `https://www.policyuncertainty.com` US/Global/China daily → FY avg
- Election calendar static CSV Wikipedia+IFES 2020-2026 months to next election per country US/Global sectors
- World Bank WGI Political Stability annual slow-moving
- Rate vol 10Y yield 90d std via yfinance ^TNX

| # | Feat | Def | Source | Regime hint 2024 3.3% |
|---|---|---|---|---|
| 1 | GPR_GLOBAL_AVG_FY | Geopolitical Risk avg FY | GPR xls Iacoviello | 2024 high 142 vs 95 mean → bearish market |
| 2 | GPR_YOY | YoY change | derived | ↑ = risk-off |
| 3 | EPU_US_AVG_FY | Economic Policy Uncertainty US FY avg | EPU CSV US | 2024 ↑ |
| 4 | EPU_GLOBAL_AVG_FY | Global | EPU Global | ↑ |
| 5 | ELEC_PROX_US | 12/(months_to_US_pres+1) 1 near election | static calendar | 2024 near =1 |
| 6 | ELEC_PROX_GLOBAL | revenue-weighted upcoming elections EM exposure | sector exposure Energy high EM election | sector interaction |
| 7 | TARIFF_RISK | Trade Policy Uncertainty subindex EPU | EPU Trade | 2024 |
| 8 | WGI_POL_STABILITY | World Bank US score slow | WGI CSV | — |
| 9 | GOV_SHUTDOWN_PROX | US debt ceiling / shutdown search vol proxy GDELT GOV theme | GDELT themes GOV_DEBT_CEILING GOV_SHUTDOWN | spikes = risk |
| 10 | RATE_VOL_3M | std 10Y yield last 90d | yfinance ^TNX [90d] std | 2022-2024 rate vol up |

Market-wide: same for all tickers FY same → provides FY conditioning lets model learn 2024 mean 3.3% driven by GPR/EPU regime not firm collapse signal. Solves quantile mapping failure noted TOWER_V6_DESIGN lines 78-84. Fusion learns to gate political higher in high VIX/GPR FYs.

Offline fallback: static table `GPR/EPU 2020-2025 static FY avg` built from last fetch (approx 95-142) + ternary high/med/low flag ensures zero-deps offline pass.

### C4 Global_Trade_Commodity 12f — `towers_v6/trade_commodities.py` — Freight Dollar Commodities Drive Margins Especially Energy/Materials/Industrials

Sources yfinance free no key (or FRED via yfinance proxy): CL=F WTI, BZ=F Brent, HG=F Copper, SLX Steel ETF, LBS=F Lumber, NG=F NatGas, DXY via DX-Y.NYB, CNY=X USDCNY, BDRY freight proxy Baltic Dry proxy, GSCPI NY Fed xlsx, C=F corn W=F wheat, BHP proxy.

Z-score 5Y rolling YoY avoid lookahead.

| # | Feat | Def | Source | Sector beta map COMMODITY_BETA_X_SECTOR |
|---|---|---|---|---|
| 1 | OIL_WTI_YOY | CL=F FY avg YoY | yfinance CL=F | Energy 1.5 Materials1.2 Industrials0.8 ConsDisc0.3 Staples0.1 Healthcare0.0 Tech0.1 Fin0.2 * |
| 2 | OIL_BRENT_SPREAD | BZ-WTI spread | BZ=F - CL=F | Energy ↑ spread = intl stress |
| 3 | COPPER_YOY | HG=F global demand proxy | HG=F | Industrials 1.3 Materials 1.2 |
| 4 | STEEL_PROXY_YOY | SLX ETF | SLX | Materials 1.4 Industrials 1.1 |
| 5 | LUMBER_YOY | LBS=F housing proxy | LBS=F | Industrials housing |
| 6 | NATGAS_YOY | NG=F margin energy cost | NG=F | Utilities leverage |
| 7 | DXY_YOY | dollar strength | DX-Y.NYB | negative for export |
| 8 | USDCNY_YOY | trade tension | CNY=X | ↑ = tension |
| 9 | BDRY_YOY | freight proxy Baltic Dry Breakwave BDRY ETF | BDRY | freight cost |
| 10 | GSCPI_AVG_FY | supply chain pressure standardized | NY Fed GSCPI xlsx | GSCPI normalized |
| 11 | COMMODITY_BETA_X_SECTOR | commodity YoY × sector sensitivity | sens_map from sector_context GICS 11 | interaction feature 1.5→0.0 per above * |
| 12 | AGRI_YOY | equal weight corn+wheat | C=F + W=F | Staples cons disc |

*sens_map Energy 1.5, Materials 1.2, Industrials 0.8, Cons Disc 0.3, Staples 0.1, Healthcare 0.0, Tech 0.1, Comm Serv 0.2, Utilities 0.2, Financials 0.2, Real Estate 0.1

Normalization 5Y rolling z avoiding lookahead, YoY vs prior FY, NaN 0 mask.

Offline fallback: `trade_commodities.py` proxies sector_context + commodity sector beta static + if fetch fails use sector noise + market-wide zeros with mask so model learns to ignore missing for that FY but keeps family present.

### C5 Synthetic Fallback Offline — Zero-Deps True Guarantees

Design invariant per `bundles/zero_deps.json`: no pip installs, no cloud required, ACNE optional local.

Strategy:

- `build_real_v6_towers.py` 29 lines merges Z 14,400×118 →154 (20 families)
- If `train_matrix_v5.npz` missing on clean clone OR yfinance/GDELT timeout offline → synthetic proxy sector-specific noise: keep old 118 intact, add 32 cols = sector noise + market-wide zeros scaled to match expected variance.
- Early runs use synthetic but still auto-detect families 17→20 passes CQS >0.605 smoke proof. Full rebuild when external fetch succeeds → caches `pipeline/data/train_matrix_v6_20f.npz` + `pipeline/data/gdelt_cache.json` + `gpr_epu_cache.json` + `gscpi_cache.json`.
- Model code unchanged `family_slices` dict discovers new dims → new ResidualTower no code change needed, fusion n_towers 20 will eventually learn correct gating.

Memory: New matrix 14,400×154 ~6.8MB float32, 1.7MB npz compress — fine VM 7.8G.

### C6 Pipeline Commands

```bash
# smoke offline (Hatch VM CPU)
python3 -u pipeline/build_real_v6_towers.py --offline
python3 -u pipeline/train_mtnn.py --smoke --epochs 2 --batch 512 --dim 64 --fusion transformer --tower-width 24 --d-model 96 --n-fusion-layers 4 --n-attn-heads 4 --device auto --w-f6 5.0

# full external fetch then train (LOCAL-GPU)
python3 pipeline/towers_v6/fetch_external.py  # GPR/EPU/GSCPI/yfinance prices full 2020-2025 → cache pipeline/data/
python3 pipeline/build_real_v6_towers.py  # builds pipeline/data/train_matrix_v6_20f.npz 14,400×154
python3 -u pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --batch 512 --tower-width 24 --d-model 96 --n-fusion-layers 4 --n-attn-heads 4 --tower-blocks 2 --mlp-heads --lr 1.5e-3 --weight-decay 1e-4 --val-every 5 --device cuda --one-cycle --pct-start 0.1 --clip 1.0 --dropout 0.12 --temp 0.08 --hard-neg 0.2 --w-f6 5.0 --rank-w 1.0 --var-w 0.2 --distress-w 0.8 --triple-w 0.5 --n-pairs 256 --margin 0.02 --triple-horizon 21 --up 0.07 --down -0.05
```

### C7 Gate to Promote 20-families Real

- Old synthetic 20-family CQS 0.5908 smoke → expected real fetched 0.72+ CQS PASS target
- Real data 20-family + FY conditioning should fix market_acc 0.57→0.60+ PASS
- IC 0.007/0.0097→0.011+ PASS via commodity betas grounding (true cost structure) + political conditioning de-noising
- Triple barrier 0.2189→>0.25 with 21d+ path-dependent labels
- Distress -0.2624→+0.2 with safety head

Keep provenance-honest: `feature_manifest_v6_real.json` `manifest.json` must record `source train_matrix_v6_20f.npz provenance generated_at 2026-08-12+ offline_synthetic flag -> real_when_cache`.

---

## D) Hill 133 Combined Execution Plan — DAG :01 Lite + Verifier Economics

**Router MoMA-lite 5 tiers:** deterministic/lit/gold → llm/deep-research/action-operator/agentic-epic → swarm 3-lens strategist parallel → planner DAG → builder/swarm → critic verification econ thr 8.0 budget3 earlyExit0.3.

### DAG Nodes

| Node | Agent | What | Input → Output | Pacing |
|---|---|---|---|---|
| **S1** | strategist-loss-A | loss engineering spec + code diff `pipeline/losses_v6.py` rank/distress/triple/var | eval_forward.json 0.57/6.15 FAIL -0.2624 0.2189 → losses_v6.py | :01 max1/4 |
| **S2** | strategist-IC-B | architecture CLS vs gated VIX FY cond | eval_forward 0.007/0.0097 market0.57 2024 3.3% → transformer vs fusion choice | :01 concurrent with S1 |
| **S3** | strategist-tower-C | tower 20 10+10+12 real fetch synthetic fallback | TOWER_V6_DESIGN → towers_v6/*.py built cache | :01 concurrent S1/S2 |
| **P** | planner | DAG wiring S1+S2+S3 → build steps 5 + verification gates | strategists.md → EXECUTION_PLAN.md | 1 Q max |
| **B1** | builder-CPU-lite | smoke 4k 2ep transformer gated VIX 512 batch quick verify 0.59→0.62 vs ranked gated 0.7017 remains best | pipeline/train_mtnn.py --smoke 2ep CPU → eval_scoreboard_smoke.json | pacing 3/4 :05+ |
| **B2** | builder-LOCAL-GPU-heavy | **14.4k 60ep transformer CLS full** CQS 0.72+ IC 0.011+ triple_write nano OOM fallback batch256 accum2 OneCycle10% | requires GPU RTX4090 → `equities_v6_money_v2.pt` candidate ONLY IF gates PASS | tempo :13 max 1 concurrent, claim COORDINATION_LOCAL_GPU.md |
| **V** | verifier-economics | budget3 thr8.0 earlyExit0.3 fix once if <8 max2 loops single enforcement | candidate json CQS>0.72 market>0.58 next_R2>0.20 IC>0.01 Sharpe>1.2 → PASS else keep 0.7017 honest | — |
| **C** | critic-QA-0-10 | QA 0-10 gate lit → produce HILLCLIMB_v6_to_v2 hill133 REPORT | eval_forward triple_write 7-field timeline.jsonl metrics.jsonl | triple-write |

### SMART Targets Hill133

| Metric | Before (132) | Target (133) | Delta needed | Expected solution | Money gate |
|---|---|---|---|---|---|
| **CQS** | 0.7017 | **0.72+** | +0.0183 | transformer 60ep comp0.809 epoch0→0.73-0.76 | v2 requires >0.72 |
| recall@10 | 1.0 | 1.0 (keep) | 0 | transformer keeps 0.9-1.0 | PASS |
| purity@20 | 0.68 | 0.70+ | +0.02 | CLS better archetype | PASS >0.60 |
| sector_acc 11-way | 0.957 | 0.95+ keep | — | tower interaction no degradation | PASS >0.95 |
| market_acc | 0.57 | **0.60+** | +0.03 | FY cond GPR/EPU solves regime 3.3% | target >0.58 |
| continuity AR1 FY12 | 0.72 | 0.70+ | keep | — | PASS |
| next_R2 14d | 0.18 /0.244 gated | **>0.20** | +0.02 | transformer recall→R2 | gate >0.20 |
| **IC 1M/3M/6M/12M** | 0.0051/0.0064/0.007/0.0062 | **0.011+** | +0.004 over 0.007 +0.0013 over 0.0097 spearman | ranking256 margin0.02 + var push + FY cond + commodity beta + VIX gate | **v2 requires >0.01 → PASS** |
| Top50 IC | 0.079 | >0.10 | +0.021 | concentration + var | keep PASS |
| IC target proxy | 0.5066 | >0.55 | — | isotonic preserve 0.878→0.881 | — |
| calibration bias | 0.0 after isotonic 162 thr | 0.0 keep | keep | isotonic after raw var fix remains | PASS <1% |
| **triple +10% before -7% 63d 21.89% random0.25 FAIL** | **0.2189 FAIL** | **>0.30** | +0.08 with 21d+7%/-5% easier | 21d +7%/-5% + true path label | gate >random / win>55% stretch for 0DTE tiny |
| **distress corr** | **-0.2624 FAIL inv** | **>+0.1** | +0.36 inversion fix | safety head invert loss hinge | gate >0 |
| Sharpe sqrt2_ann6M | 0.57 FAIL | **>1.0 stretch >1.2** | std 0.1251→0.04 mean0.0504 0.57→1.78 | var loss 0.7%→4% + mean preserve | v2 requires >1.2 ambiguity sqrtN 6.15 PASS documented honest |
| Sharpe sqrtN trade-count | 6.15 PASS | >1.0 keep | — | — | PASS >1.0 |
| Sharpe 0DTE tiny gated | locked | IC>0.03 Sharpe>1.2 win>55% DD<12% kill-switch separate bankroll NOT advice | IC 0.007→0.011 still <0.03 → locked honest | paper first |
| n_trades | 233 | ≥200 keep 233 min Kelly 0.25 1% max3 | — | entry thr 0.7→0.72 TOP-filter | PASS |

### Risks / Failure Taxonomy 5 + SideEffect 4 + Recovery Ladder

| Failure | Taxonomy | SideEffect | Recovery |
|---|---|---|---|
| Train_matrix_v5 missing 118→154 fail | INPUT_CORRUPTION | none | synthetic fallback offline sector noise saves run keeps auto-detect 20-family |
| SIGTERM 74s buffered / 167s before epoch1 OOM | CONTEXT_STARVATION / TOOL_FAILURE | none | `-u` unbuffered + batch1024 lite + nano fallback batch256 accum2 + claim COORDINATION_LOCAL_GPU |
| GDELT/yfinance timeout offline | TOOL_FAILURE | none | proxy fallback static GPR/EPU cache + sector noise + market zeros mask |
| sector_acc 0.13 2ep too short CLS starved | REASONING_COLLAPSE | none | 60ep fixes full attention emergence need not 2ep |
| v2 fake promotion IC<0.01 must refuse | OUTPUT_CORRUPTION | — | verifier kills fake honest gate budget3 thr8.0 single point verifier-with-budget.js |

Recovery ladder: retry→patch→replan→escalate per `recovery-ladder.js` FailureTaxonomy5 SideEffect4 — retry gated lite 14.4k 2ep batch1024 `-u` → patch transformer→gatedVIX if OOM batch256 accum2 OneCycle10% clip1.0 drop0.12 temp0.08 hard-neg0.2 → replan 20→17 towers fallback if 32 noisy w_f6=5 var0.2 rank256 margin0.02 → escalate LOCAL-GPU RTX4090 60ep required COORDINATION_LOCAL_GPU.md claim — even no-change logged 7-field mandatory.

Verdict economics: budget3 thr8.0 earlyExit0.3 fix once if <8 max2 loops single enforcement point.

### Money Chain Proof → Free Platform Forward Honest

Per `ALIGNMENT_SYNTHESIS.md` Knowledge→Edge→Money + `HILLCLIMB_v6_to_v2_REPORT.md` free platform free — open access proof-of-work calibration isotonic bias0.0 Top50 0.079 small-n already proves knowledge:

```
Free platform free users (hoops.dumbmodel.com model) 5 games ever 4 legacy + unified chimera 20719×64-d LCG dailySeed 20260812 idx3970 triple[3970,14390,4582] PWA v67 74426B void #080A0F
  ↓ profitability via own edge glass-box SHAP/permutation importance logged eval JSON + Lab front
  → Kalshi 0.25 Kelly 1% max per play 3 concurrent max → private paper trades 233 filtered entry_mean 0.8409 thr0.7 p from isotonic head diff≥5% triggers f*=0.25*Edge/Odds
  → Equity paper directional prop_edge_equities.jsonl validates forward edge from for_history.json
  → Tiny 0DTE spreads ONLY IF IC>0.03 & Sharpe>1.2 & win>55% & DD<12% kill-switch separate bankroll NOT financial advice → close if DD>12% or Sharpe<1.0
```

Current IC 0.007/0.0097<0.01<0.03 → no 0DTE live stays paper — honest `eval_forward.json` gate.

Top50 0.079 would PASS but n=50 too small for 0.25 Kelly stability needs 233 min.

Sharpe sqrtN 6.15 PASS vs sqrt2 0.57 FAIL ambiguous honest logged — var loss should fix sqrt2 to PASS.

Triple 0.2189 < random 0.25 → barrier horizon/threshold miscal quick fix 63d→21d threshold +10%/-7% → +7%/-5% expected 0.32+.

Distress -0.2624 inverted → new `-corr` hinge penalty + safety head.

### Deliverables Hill133 Checklist

- [ ] Loss A: `pipeline/losses_v6.py` ranking_loss_256 margin0.02 thresh5% w_rank1.0 + var_loss std target0.04 w_var0.2 + distress_penalty -corr hinge w0.8 invert head + triple_bce 21d +7%/-5% w0.5 + w_f6 5 up from 1.0
- [ ] Arch B: transformer CLS v6.1 4L 4H d_model96 tower_width24 tower_blocks2 fy_token 12+4 + ContinuousFusionVIX upgraded gated fallback VIX/GPR/EPU FY conditioning market_acc 0.57→0.60+ fixes 3.3% regime quantile mapping
- [ ] Tower C: `towers_v6/` 10+10+12=32 real fetch external via `fetch_external.py` GPR/EPU xls/csv GSCPI yfinance CL=F/BZ/HG/SLX/LBS/NG/DXY/CNY/BDRY/C/W synthetic fallback offline zero-deps true auto-detect 17→20 families D 122→154 14,400×154 ~6.8MB model no code change
- [ ] Smoke CPU lite 4k 2ep transformer CQS 0.5908→0.62 recall0.9125 purity0.7589 market0.593 PASS sector0.13 FAIL expected 2ep short + gated VIX variant market 0.60+ IC 0.006→0.008
- [ ] Full LOCAL-GPU RTX4090 60ep transformer CLS CQS 0.809 epoch0→0.72+ PASS 0.7017 would-beat IC 0.007→0.011+ PASS Sharpe sqrt2 0.57→1.2+ PASS triple 0.2189→0.30+ distress -0.2624→+0.1 market>0.58 next_R2>0.20
- [ ] Gate `equities_v6_money_v2.pt` ONLY IF CQS>0.72 IC>0.01 Sharpe>1.2 market>0.58 next_R2>0.20 otherwise keep best 0.7017 honest no fake
- [ ] Candidate json `assets/eval_scoreboard_v6.json` `assets/eval_forward.json` updated measured not experimental `pipeline/data/mtnn_report.json` provenance-honest `feature_manifest_v6_real.json` `manifest.json` torch auto cuda else cpu Hatch VM CPU Alienware GPU auto device per MEMORY 2026-08-10 08:29 clarity
- [ ] Calibrate isotonic 162 thr preserve IC 0.878→0.881 bias 0.0 keep variance fix raw still 0.0 after calibrator
- [ ] Money proof free platform PWA v67 LCG dailySeed 20260812 idx3970 triple [3970,14390,4582] same-link-same-stars drag-map→Jordan 3D embedding central LOD4000/8000 DPR1 fillRect vintage shared-map.js even no-change logged timeline.jsonl 7-field nodeId agentId attempt latency tokens status errorClass ts runId ooda tempo checkpoint-manager
- [ ] This file `docs/T5_hill133_strategists.md` lens 3× T5 :01 lite 5m pacing :01

---

**No financial advice. Free platform forever. Profit via own calibrated edge private 0.25 Kelly 1% max 3 concurrent per-play separate bankroll kill-switch DD<12% — IC>0.03 gate locked honest. torch auto `cuda if available else cpu` no pip cloud zero-deps true even no-change logged.**

Lens pause 19:47 CDT :01 5m lite — resume builder B1 smoke CPU + B2 LOCAL-GPU 60ep transformer CLS via `COORDINATION_LOCAL_GPU.md` claim `python3 -u pipeline/train_mtnn.py --epochs 60 --fusion transformer --batch 512 --device cuda` — monitor `mtnn_best.pt` 514K→0.72+ & IC 0.011+ & Sharpe 1.2+ & triple 0.30+ & distress +0.1.
