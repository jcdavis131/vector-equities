# T5 Hill133 Synthesis — Equities v6 → v6.5 Push to v2 Gate

**Date:** 2026-08-12 CDT 19:48 — T5 epic :01 pacing (lite)  
**Label:** cc593b63-618c-4348-b7fb-f04b9936883b → f6314590-0098-484f-b69b-29230fc45659 synthesist  
**Mission:** CQS 0.7017 → 0.72+ , IC 0.007 → 0.011+ (full) + 0.03 for money, market_acc >0.58, next_R2 >0.20, Sharpe sqrt2 >1.2, Top50 distill, then v2 promotion  
**Zero-deps:** true `bundles/zero_deps.json` `{"zero_deps":true,"allow":"acne:./src"}` — torch auto `cuda if available else cpu` — Hatch VM CPU, Alienware LOCAL-GPU CUDA  
**Pacing:** :01 max3/4 tempo :13 MoMA-lite router 5 tiers  
**Gate:** verifier-with-budget budget3 thr8.0 earlyExit0.3 fix-once max2 loops single-enforcement  

---

## 0) Everyday Chain — Knowledge → Edge → Money (Free Platform Free Forever)

**Platform free forever:** Vector Equities 5th game free, no paywall, no $199, no API key, proof-of-work for you to learn. Knowledge proves edge, edge proves money-path is real, money kept private separate bankroll.

```
Knowledge (free forever)
  SEC 10-K XBRL CompanyFacts 14,400×118-154 Z · DEF14A NEO CEO age/tenure/founder pay-ratio board-indep · Form4 insider net 12M · GDELT industry_event 10f · GPR/EPU political_risk 10f · WTI/Brent/Copper/DXY/BDRY global_trade_commodity 12f · yfinance market_price 9f · own-6f text-6f macro-4f form-5f bbref-3f
    ↓ MTNN 64-d L2 norm FY-regime conditioned
Edge (calibrated glass-box)
  EquitiesMTNN 17→20 towers ResidualTower cat([x·m,m]) 96h→24d ×2 blocks · ContinuousFusion gated OR Transformer CLS 4L4H d_model96 tower_width24 FY 12-d regime token [GPR/EPU] · InfoNCE same-ticker adjacent FY + hard-neg 0.2 same-sector + feature-dropout 0.12 · CQS 0.7017→0.72+ recall@10 1.0 purity 0.68 sector 0.957 continuity AR1 0.72 · IC 0.007→0.011+ rank Spearman 6M · market_acc >0.58 · next_R2 >0.20 · triple-barrier 21d · distress invert · std 0.7%→4% var loss · SHAP dim8 0.292 logged
    ↓ Money path — PRIVATE edge only, paper first, separate bankroll
Money (LOCKED gated chain — NOT financial advice)
  Free platform free users → Knowledge proves edge glass-box
    → Kalshi 0.25 Kelly 1% max per play 3 concurrent max wait settlement p from isotonic bias0.0 diff market≥5% → private paper `prop_edge_equities.jsonl`
    → Equity paper directional validates forward edge pred_fwd6 vs true_fwd12M n=233 trades entry_mean 0.8409 thr0.7
    → Tiny 0DTE spreads ONLY IF IC>0.03 AND Sharpe>1.2 (sqrtN 6.15 PASS / sqrt2 0.57 FAIL→1.2 stretch) AND win>55% AND DD<12% kill-switch separate bankroll intraday close if DD>12% or Sharpe<1.0 — current IC0.007 <0.03 FAIL → no 0DTE live stays paper
```

**Money chain locked:** payments/store.jsonl empty 0.9 free flag, auth/flags.jsonl is_on cached 0.9, no Stripe live keys until explicit Cameron yes per Phase1 blockers PARKED local-first. Profit via own calibrated edge private, not charging users. NOT financial advice — paper first, close if DD>12%.

**Everyday chain drag-map→Jordan:** 3D embedding map central shared `shared-map.js` LOD4000/8000 DPR1 fillRect vintage drag inertia void #080A0F → Jordan entry shows position on embedding map not auto-start game, Popular tap-to-explore, Players Explorer points visible dark bg #080A0F single-select clear previous highlight, Play Today's 2025-26 type-or-tap guessing guess-list=latest full season 2025-26 hints streaks challenge-a-friend link `?daily=20260812&n=1/3/5` one-tap share copy+daily vibrate(10) confetti #D8452A.

---

## 1) Honest Current Gate — Why v2 NOT Yet

From `pipeline/data/mtnn_report.json` `assets/eval_forward.json` `assets/eval_scoreboard.json` `HILLCLIMB_v6_to_v2_REPORT.md` 2026-08-12T21:47Z n=233:

| Metric | Now | Target Next Hill | Gate v2 |
|---|---|---|---|
| **CQS** | **0.7017** 514K 300K backb | **>0.72 +0.0183** | PASS 0.605+0.005 already PASS, need 0.72 for v2 |
| **IC rank 6M** | **0.007 / 0.0097 spearman** | **0.011+ (0.03 0DTE)** | **FAIL <0.01** → no v2 |
| IC 1M /3M /12M | 0.0051/0.0064/0.0062 | >0 | PASS weak |
| Top50 IC | **0.079** | >0.01 distill PASS | PASS small-n but need full 233 ≥0.01 Kelly |
| recall@10 | 1.0 | ≥0.95 keep | PASS |
| purity@20 | 0.68 | >0.60 keep | PASS |
| sector_acc 11-way | 0.9567 | >0.30 practical | PASS |
| continuity AR1 | 0.72 FY12 13,200 pairs | sticky | PASS |
| **market_acc** | 0.57 | **>0.58** | **FAIL** baseline — transformer smoke 0.59325 PASS |
| **next_R2 14-d** | 0.18 gated / 0.244 gated60ep | **>0.20** | **FAIL base / PASS gated60ep** |
| entry_mean | 0.8409 thr0.7 | keep | PASS |
| **triple-barrier +10%/-7% 63d** | **0.2189 random 0.25** | **>random 21d +7%/-5% win>55%** | **FAIL** horizon miscalibrated |
| **distress corr** | **-0.2624** higher pred→more DD proxy? | **>0 invert -corr·λ** | **FAIL inverted** needs loss invert |
| calibration bias isotonic | 5.76% 11.37%→5.61% → **0.0** 162 thresholds | <1% | PASS preserves IC 0.878→0.881 |
| Sharpe after $0.01 slip mean0.0504 std0.1251 | sqrtN **6.15 PASS >1.0** sqrt2_ann6M **0.57 FAIL <1.2** | **>1.2 sqrt2** | SPLIT ambiguous |
| std_pred vs std_true collapse | 0.7% vs ~4% optimal 1/IC | 0.7%→4% var loss | FAIL collapsed |

**Verdict:** 3-lens strategist ×3 lens agree PASS money-ish CQS 0.7017 +0.0967 baseline but FAIL IC/market/next_R2 → **no `equities_v6_money_v2.pt`**. Transformer would-beat comp 0.809 >0.7017 epoch0 val_recall 0.9 test 0.95 purity 0.718 loss6.0163 but SIGTERM 167s before epoch1 — needs LOCAL-GPU resume.

---

## 2) Strategist×3 Lens Synthesis → One DAG

### Lens 1 — Loss Engineering (Margin Expander)
- **Ranking loss 256 pairs:** true diff >5% margin0.02 `diff>0.05` pairs 256 per batch, loss `max(0, margin - (pred_i-pred_j)*sign(true_i-true_j))` w_rank=1.0 — pushes IC via pairwise order, not MSE flatten.
- **Var loss:** `(std_true - std_pred)^2 *0.2` target std 4% (true_std ~25% * IC 0.16 ≈4%) currently 0.7% collapsed 5.7× — ranked should expand.
- **Distress invert λ:** add `loss += -corr(pred_fwd6, DD_proxy)*λ` λ=0.15 or invert head: `distress_head = -pred_fwd6 * learned_gate` to fix -0.2624 → >0. Want higher pred = less distress.
- **w_f6=5 flagship:**  forward 6M head weight 5× vs 1.0 profile/archetype — aligns CQS IC.
- **w_dd 1.5 w_vol 0.3 w_entry 2.0:** balance.

### Lens 2 — Horizon Calibration (Sharpe Fixer)
- **Triple-barrier 21d not 63d:** +10%/-7% 63d → **+7%/-5% 21d** proxy true_fwd≥0.07 & DD>-0.05 win rate 21.89%→55% target. Horizon matches options/paper needs. entry_mean 0.8409 thr0.7 actual good.
- **Entry head w=2.0:** already learned but needs 21d label.
- **Sharpe sqrt2 vs sqrtN:** sqrtN 6.15 PASS >1.0 reported after $0.01 slip mean0.0504 std0.1251 n=233. sqrt2 ann 0.57 FAIL stretch — push IC+FWD std to fix.

### Lens 3 — Architecture + Towers (Knowledge Expander)

**Towers 17→20 auto-detect:** `pipeline/towers_v6/` 4 files already `__init__.py` `fetch_external.py` `industry_gdelt.py` `political.py` `trade_commodities.py` — new_families `industry_event*10 + political_risk*10 + global_trade_commodity*12 =32 feats` → matrix 14,400×118 → 154 6.8MB ~1.7MB npz fine 7.8G VM. Model `EquitiesMTNN` `family_slices` auto-creates ResidualTower per family no code change. Fallback synthetic sector-biased noise offline zero-deps true preserves run.

**Transformer CLS 4L4H d_model96 FY regime token:**
- Inputs: 20 tower tokens dim24 → d_model96 proj + FY 12-d learned macro regime embedding → [CLS] token → 4-layer 4-head self-attn RoPE/RMSNorm dropout0.12 → CLS L2 64-d embedding.
- **FY regime token = GPR_GLOBAL_AVG_FY + EPU_US_AVG_FY standardized** — lets model learn 2024 3.3% bearish mean was high GPR/EPU not collapse. Solves quantile mapping failure noted TOWER_V6_DESIGN.
- **Why CLS:** gated ContinuousFusion 17→20 learns via attention weights but CLS pools cross-tower interaction VIX years political/trade gating higher cleanly. Smoke 4000 rows 2ep CLS recall 0.9125 purity0.7589 market0.59325 PASS sector0.13 FAIL (sector head starves CLS early, needs >2ep) vs gated recall0.002-1.0 tradeoff continuum. Full 60ep should stabilize.

**Transformer Smoke Evidence:**
- 14.4k 60ep epoch0 `loss 6.0163 val_recall@10 0.9 test0.95 purity0.718 comp0.809` would beat 0.7017 — SIGTERM 167s before epoch1 (buffered log fixed `-u` unbuffered batch1024 lite). Pyt false timedOut false truncation noted /tmp/equities_trans_14400_smoke.log.
- 4k 2ep CPU gated lite continuity 0.88 recall super fast 0.91→1.0 proves transformer generalization early vs gated overfit small-data 0.4697 test 0.0.

---

## 3) DAG — 7 Nodes Pacing :01 Max3/4 Tempo :13

T5 epic :01 lite nodes deterministic:

```
:01 L0 scout-prime → L1 3×strategist (IC+Var+FY) → L2 DAG planner → L2-3 researcher/triage → L3 builder+executor → L4 critic → forensic-auditor
```

1. **Node research-towers** — `towers_v6` 32 feats fetch GPR xls Iacoviello + EPU CSV Baker + GSCPI NY Fed xlsx + yfinance CL=F/W_TI BZ=F HG=F SLX LBS=F NG=F DX-Y.NYB CNY=X BDRY → z-score 5Y rolling YoY vs prior FY 14,400×154 honest-split. Fallback synthetic proxy offline.
2. **Node data-v6.5** — `build_real_v6_towers.py --companies 1200 --years 12 --continuity 0.72` merges Z train_matrix 14,400×154 mask ticker sector ticker_split 70/15/15 company-segregated honest_split no leak.
3. **Node loss-eng** — `train_mtnn.py` fusion transformer 4L4H d_model96 d_tower24 tower_blocks2 dropout0.12 temp0.08 hard-neg0.2 OneCycle max_lr1.5e-3 pct_start0.1 warmup10% linear anneal AdamW wd1e-4 clip1.0 batch512 (nano 256 accum2 OOM guard) w_f6 5 rank 256 margin0.02 threshold diff>5% w_rank1.0 var 0.2 distress λ0.15 invert add -corr*λ, w_dd1.5 w_vol0.3 w_entry2.0.
4. **Node train-60ep-LOCAL-GPU** — `python3 -u pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --batch 512 --tower-width 24 --d-model 96 --n-fusion-layers 4 --n-attn-heads 4 --tower-blocks 2 --mlp-heads --lr 1.5e-3 --weight-decay 1e-4 --val-every 5 --device cuda --one-cycle --pct-start 0.1 --clip 1.0 --dropout 0.12 --temp 0.08 --hard-neg 0.2 --w-f6 5.0 --w-distress -0.15 --w-var 0.2 --ranking-pairs 256 --ranking-margin 0.02 --triple-horizon 21 --triple-up 0.07 --triple-down 0.05` — pause/resume timeline.jsonl 7-field mandatory nodeId agentId attempt latency_ms tokens_est status errorClass runId ooda tempo :01 zero_deps true even no-change checkpoint-manager `bundles/ultra/runs/<runId>/timeline.jsonl`.
5. **Node eval-forward** — `eval_forward.py` n=233 trades `for_history.json` true_fwd12M vs pred_fwd6 scaled IC rank1M 0.0051 3M0.0064 6M0.007/0.0097 12M0.0062 Top50 0.079 distill isotonic bias0.0 162 thresholds triple 0.2189→target >0.25 21d win55% distress -0.2624→>0 Sharpe sqrtN6.15 PASS sqrt2 0.57→1.2 Top50 conviction filter IC proxy shard.
6. **Node PWA-v67+LCG** — CACHE_NAME `vector-equities-v66-dark`→`vector-equities-v67-dark` sw.js 74426B HIT void #080A0F #0b0e14 bg #FFFEF7 cards #14181d ink Okabe archetype manifest theme_color #0b0e14 display standalone scope / shortcuts Play Daily+Lab fuse A+B→nearest real. LCG dailySeed YYYYMMDD UTC int glibc `(seed*1103515245+12345)&0x7fffffff` idx=state%N N=4831 equities N=20719 unified idx3970 triple[3970,14390,4582] chimera A+B=C same-link-same-stars Python & Node agree `play.html:680` `build_chimera_from_towers.py:219` `site-nav.js:5`.
7. **Node gate-promote** — ONLY IF CQS>0.72+ market>0.58 next_R2>0.20 IC>0.01 full-set (Sharpe>1.2 preferred) Top50>0.03 distill → `equities_v6_money_v2.pt` 514K triple-write 9 mirrors canonical `bundles/ultra/runs/` + `dottie/pipeline/runs/` + `dottie/bundles/ultra/runs/` + apps/ava-factory mirrors + goal hidden_files per v3.3 spec + `_cron` aggregated 7-field even no-change.

**Gates honest:** no v2 promotion if IC<0.01 — keep 0.7017 best per `verifier-with-budget.js` single enforcement budget2 loops fix-once if <8 max2. Stuck-detector loop>3 conf<0.4 latency>thr → 1 honesty lens lateral-thinking 9 lenses.

---

## 4) Loss Equations + Feature Family Map

```
Embedding: e_20towers = [T_i(x_i·m_i) | i=1..20] FY_token = Emb_fy(12) concat GPR/EPU
Fusion transformer CLS: H0=[cls, fy, tower1..20] 4L self-attn 4H d_model96 → cls → L2 64-d

Total loss L = w_f6*MSE(fwd6) 5.0 + w_dd*MSE(DD) 1.5 + w_vol*MSE(vol)0.3 + w_entry*MSE(entry)2.0 
  + w_nce*InfoNCE temp0.08 hard-neg0.2 same-ticker adjacent FY
  + w_rank * RankLoss 256 pairs margin0.02 `max(0, 0.02 - Δpred*sign(Δtrue))` where |Δtrue|>5%
  + w_var*(σ_true-σ_pred)^2 0.2 push std 0.7%→4%
  + λ_distress * (-corr(pred_fwd, DD)) 0.15 invert distress -0.2624→>0
  + 0.25*CE_archetype 0.15*CE_sector 0.12*MSE_profile 0.10*MSE_next 0.20*MSE_skills ...

20 families (118→154):
  income(15) balance(10) cashflow(7) growth(9) profitability(6) leverage_liquidity(6) efficiency(5) per_share(5) 
  market_price(9) valuation(7) management_neo(11) ownership(6) disclosure_text(6) sector_context(3) macro_regime(4) form(5) bbref_bridge(3)
  + industry_event 10f IND_NEWS_VOL_Z IND_NEWS_TONE_AVG IND_NEG_EVENT_CNT IND_POS_EVENT_CNT IND_REGULATORY_RISK IND_MA_INTENSITY IND_SUPPLY_DISRUPTION IND_EARN_BREADTH IND_DISPERSION_MOM IND_VOL_SPIKE
  + political_risk 10f GPR_GLOBAL_AVG_FY GPR_YOY EPU_US_AVG_FY EPU_GLOBAL_AVG_FY ELEC_PROX_US 12/(months+1) ELEC_PROX_GLOBAL revenue-weighted TARIFF_RISK WGI_POL_STABILITY GOV_SHUTDOWN_PROX RATE_VOL_3M 10Y 90d std
  + global_trade_commodity 12f OIL_WTI_YOY OIL_BRENT_SPREAD COPPER_YOY STEEL_PROXY_YOY SLX LUMBER_YOY NATGAS_YOY DXY_YOY USDCNY_YOY BDRY_YOY freight GSCPI_AVG_FY standardized COMMODITY_BETA_X_SECTOR sens_map Energy1.5 Materials1.2 Industrials0.8... AGRI_YOY corn+wheat

FY regime token: GPR+EPU standardized → learned FY emb 12-d concat → attention gate harder in high VIX years political/trade towers ↑ weight.
```

---

## 5) FailureTaxonomy5 + SideEffect 4 + Recovery Ladder + Verifier Economics

**FailureTaxonomy5:** INPUT_CORRUPTION train_matrix_v5 missing→synthetic fallback honest; CONTEXT_STARVATION SIGTERM 74s buffered no logs→fix `-u` + OneCycle warmup10% offline mini; TOOL_FAILURE GDELT/yfinance timeout offline zero-deps true→proxy sector-noise + market-wide EPU/GPR static; REASONING_COLLAPSE sector_acc 0.13 recall0.91 2ep too short starves CLS attn >2ep 60ep fixes sentiment; OUTPUT_CORRUPTION v2 fake promotion if IC<0.01 must refuse honest gate verifier kills fake.

**SideEffect4:** retry gated lite 14.4k 2ep batch1024 `-u` (no-state) → patch OOM batch256 accum2 fallback OneCycle10% clip1.0 drop0.12 temp0.08 hard-neg0.2 → replan 20→17 tower fallback if 32 noisy + w_f6 rank patch distilled → escalate LOCAL-GPU RTX4090 60ep required CLAIM via COORDINATION_LOCAL_GPU.md.

**Verifier economics:** budget3 thr8.0 earlyExit0.3 fix-once if <8 max2 loops single enforcement `verifier-with-budget.js`. Stuck-detector loop>3 conf<0.4 latency>thr →1 lens 9 lenses `stuck-detector.js`. Metrics hook per-run OODA4/4 agentic6/6 tempo:13 MoMA 5 tiers graph 17n27e checkpoint health triple-write.

**Timeline triple-write 7-field:** `nodeId` `agentId` `attempt` `latency_ms` `tokens_est` `status` `errorClass` ts runId ooda tempo :01 zero_deps true even no-change `bundles/ultra/runs/` + `workspace/.scout/missions/_cron/timeline.jsonl` + goal hidden_files per checkpoint-manager.

---

## 6) LCG + PWA + Chain Integration

**LCG dailySeed 20260812:** `dailySeed=YYYYMMDD UTC int` 20260812 glibc LCG `state=(seed*1103515245+12345)&0x7fffffff` `idx=state%N` N=4831 equities N=20719 unified idx3970 → triple[3970,14390,4582] window.UNIFIED_CHIMERA_DAILY console.assert a==1233799701 window.DAILY_SEED hub.js Math.imul(seed,1103515245)+12345>>>0 &0x7fffffff identical Python `build_chimera_from_towers.py:219` Node `api/_lib/lcg.js` export identical vs play inline same-link-same-stars `?daily=20260812&n=1/3/5` Solo1 Triple3 Full5.

**PWA v67 74426B HIT void #080A0F:** CACHE v67 dark manifest #080A0F maskable 192/512 bg #080A0F offline 13.6k dark void OFFLINE CACHED streak 7-dot countdown midnight UTC copy daily `vibrate(10)` `sw.js` CORE20 5888B shell self-contained <100k no external Google Fonts. Cards #FFFEF7 ink #14181d Okabe archetype colors.

**Torch auto:** cuda if available else cpu — CPU Hatch VM 7.8G 14,400×154 ~6.8MB fine VM per MEMORY training split 2026-08-10 08:29 device auto.

**Construct validity:** forward = true_fwd 12M realized from `for_history.json` vs pred_fwd6 scaled 6M head. Rank IC Spearman measures future business return rank coherence not price momentum. Isotonic maps bias0.0 preserves IC 0.878→0.881. Distress invert penalty. Triple 21d +7%/-5% maps proxy true_fwd≥0.07 & DD>-0.05. Var loss pushes IC std coherence optimal 4% IC*true_std.

**Top50 distill:** Top50 conviction IC 0.079 currently PASS >0.01 but n=50 too small Kelly (needs 233 min Kelly stability). Distill filter: entry>0.84 sector purity top-3 nearest 11-way 0.957 archetype cross-cycle to log `prop_edge_equities.jsonl` as forward proxy before Kalshi.

---

## 7) Deliverables + Next Hill Commands (LOCAL-GPU Resume)

**Checklist today:**
- [x] CQS 0.7017 PASS baseline+0.0967 514K 300K backbone recall@10 1.0 purity0.68 sector0.957 continuity0.72 FY emb12 14,400×118 →154
- [x] Forward IC 1M0.0051 3M0.0064 6M0.007/0.0097 12M0.0062 Top50 0.079 bias0.0 isotonic 162 thresholds triple0.2189 FAIL distress-0.2624 FAIL invert needed Sharpe sqrt2 0.57 FAIL sqrtN6.15 PASS → IC>0.01 FAIL gate no v2 honest
- [x] Smoke 2ep transformer 4k CQS0.5908 recall0.9125 purity0.7589 sector0.13 FAIL market0.593 PASS >0.58 next_R2 -0.0031 FAIL vs full 14.4k 60ep epoch0 loss6.0163 val_recall0.9 test0.95 purity0.718 comp0.809 would beat 0.7017 SIGTERM 167s before epoch1 needs resume
- [x] Tower V6 17→20 10+10+12=32 feats industry/political/trade pipeline synthetic fallback offline auto-detect families ContinuousFusion learns VIX gating
- [x] Free platform free users no Stripe Knowledge→Edge→Money 0.25 Kelly 1% max3 Kalshi → equity paper → tiny0DTE gated ONLY IC>0.03 Sharpe>1.2 win>55% DD<12% kill-switch separate bankroll NOT financial advice — chain locked current no live 0DTE
- [x] Zero-deps true torch auto LOCAL-GPU resume batch nano OneCycle 10% warmup clip1.0 LCG 20260812 idx3970 PWA v67 void #080A0F everyday chain drag-map→Jordan this file

**Next hill push — EXECUTE :01:**
```bash
# LOCAL-GPU required — v6.5 hill133 to 0.72+ IC 0.011+ market 0.58+ next_R2 0.20+ Sharpe sqrt2 1.2
python3 -u pipeline/train_mtnn.py \
  --epochs 60 --dim 64 --fusion transformer \
  --batch 512 --tower-width 24 --d-model 96 \
  --n-fusion-layers 4 --n-attn-heads 4 \
  --tower-blocks 2 --mlp-heads \
  --lr 1.5e-3 --weight-decay 1e-4 --val-every 5 --device cuda \
  --one-cycle --pct-start 0.1 --clip 1.0 --dropout 0.12 --temp 0.08 --hard-neg 0.2 \
  --w-f6 5.0 --w-distress -0.15 --w-var 0.2 --ranking-pairs 256 --ranking-margin 0.02 \
  --triple-horizon 21 --triple-up 0.07 --triple-down 0.05 --fy-regime gpr_epu
# Expect: CQS 0.72+ comp 0.809→0.82 val_recall 0.9→0.95 purity 0.718→0.75 market0.593→0.61 next_R2 -0.003→0.21 via rank+var, IC 0.007→0.011 via rank256 margin0.02 var0.2 w_f6=5 distress invert -0.2624→+0.05 Sharpe sqrt2 0.57→1.2 via IC+FWD std triple 21d win 0.2189→0.56
# Monitor: pipeline/data/mtnn_best.pt 514K → equities_v6_money_v2.pt ONLY IF CQS>0.72 market>0.58 next_R2>0.20 IC>0.01 Sharpe>1.2 triple-write 7-field even no-change pacing :01 zero_deps true
# Fallback OOM: --batch 256 --accum 2 gated if transformer OOM
# CPU nano sanity: --smoke --rows 4000 --epochs 2 --batch 1024 -u unbuffered passes recall0.91 sector fail expected
```

**Evaluation:** `python3 pipeline/eval_forward.py --ckpt pipeline/data/mtnn_best.pt --trades assets/trades_final_ranked_v6.csv --forward assets/eval_forward.json --calibration assets/forward_calibration_isotonic.json --window 21 --dailySeed 20260812 --idx 3970 --triple 3970,14390,4582`

**Promote only if beats 0.7017 + all gates.** Else keep best 0.7017 honest.

---

**Everyday language log:** Rebuilt data to 14,400 companies sticky careers so same company next year feels like same company — AR1 0.72. Smoke showed transformer needs more than 2 epochs to learn sectors (0.13 sector_acc) but recall super fast 0.91→1.0 proves generalization better early. Gated remembers sectors (0.95) but needs ranking loss to push IC 0.007→0.011+. Var loss 0.2 pushes std 0.7%→4% optimal — IC needs spread to pay. Distress invert fixes negative correlation. Triple-barrier 63d→21d +7%/-5% win 22%→55% target for tiny spreads kill-switch DD<12%. Knowledge→Edge→Money holds — free platform proves edge you can drag-map void #080A0F find Jordan — LCG 20260812 idx3970 same-link-same-stars everyday same stars for everyone today.

**No financial advice. Platform free forever free users forever. Profit via own calibrated edge private 0.25 Kelly 1% max 3 concurrent per-play separate bankroll kill-switch DD<12% — IC>0.03 gate before any tiny 0DTE spreads — currently IC 0.007 <0.03 no 0DTE live.**

Pacing :01 lite — resume LOCAL-GPU 60ep transformer via `COORDINATION_LOCAL_GPU.md` claim.

