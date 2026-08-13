# T5 Hill 133 — Planner DAG7-MoMA-lite OODA4/4

**Date:** 2026-08-12 19:48 CDT  
**Label:** cc593b63-618c-4348-b7fb-f04b9936883b  
**Epic:** equities v6 next hill 133  
**Parent:** hill132 CQS0.7017 PASS v6 money best  
**Zero-deps:** true `bundles/zero_deps.json {"zero_deps":true,"allow":"acne:./src"}` torch auto `cuda if available else cpu` no pip no cloud ACNE optional local

## Input — 14,400 FYs 122→154 Feats 17→20 Towers

- N=14,400 FYs, FY emb 12-d, FY pairs 13,200 continuity AR1 0.72
- D=122 → 154 (+32) via pipeline/towers_v6/__init__.py
- Towers 17→20 new 3: industry_event 10f, political_risk 10f, global_trade_commodity 12f
- Matrix: train_matrix_v6_20f.npz 14.4k×154 ~6.8MB npz ~1.7MB 7.8G fine Z 14400×118→154
- Memory auto-detect families EquitiesMTNN family_slices → new ResidualTower no code change ContinuousFusion gated n_towers 17→20 → learns VIX weighting

### Tower V6 Design (from TOWER_V6_DESIGN.md)
- industry_event 10f IND_NEWS_VOL_Z TONE_AVG NEG_CNT POS_CNT REGULATORY_RISK MA_INTENSITY SUPPLY_DISRUPTION EARN_BREADTH DISPERSION_MOM VOL_SPIKE — GDELT 2.0 Doc + GKG + SEC 8-K Item 1.01/2.01/7.01 sector/year
- political_risk 10f GPR_GLOBAL_AVG_FY GPR_YOY EPU_US_AVG_FY EPU_GLOBAL_AVG_FY ELEC_PROX_US 12/(months+1) ELEC_PROX_GLOBAL revenue-weighted TARIFF_RISK WGI_POL_STABILITY GOV_SHUTDOWN_PROX GDELT GOV RATE_VOL_3M — GPR xls Iacoviello + EPU CSV Baker-Bloom-Davis + election static CSV + WGI slow; market-wide FY conditioning solves 2024 bearish mean 3.3% regime not collapse signal
- trade_commodity 12f OIL_WTI_YOY CL=F BRENT_SPREAD COPPER_YOY HG=F STEEL SLX LUMBER LBS=F NATGAS NG=F DXY_YOY USDCNY CNY=X BDRY_YOY GSCPI_AVG_FY NY Fed xlsx COMMODITY_BETA_X_SECTOR sens_map Energy1.5 Materials1.2 Industrials0.8 etc AGRI_YOY corn+wheat — YoY vs prior FY 5Y rolling z-score avoid lookahead fill NaN 0 mask

## Baseline v6 Money Best — CQS0.7017 PASS

- file equities_v6_money_best.pt 514K (300K backbone) d_tower24 d_hidden96 tower_blocks2 fy_emb12 d_emb64 dim64 17→20 towers transformer CLS 4L 4H d_model96
- CQS 0.7016666666666668 vs 0.605 +0.005 → PASS +0.0967
- recall@10 1.0 PASS, purity@20 0.68 PASS >0.60, sector_acc 0.9567 PASS >0.30 practical >0.95, continuity 0.72 PASS FY12 14400, market_acc 0.57 FAIL target 0.58, next_R2 0.18 FAIL baseline / 0.244 PASS gated60ep, ticker split 70/15/15 honest_split true no leak
- Composite 0.5*recall+0.5*purity fixes early restore bug
- Training: OneCycle max_lr1.5e-3 pct_start0.1 warmup10% linear anneal AdamW weight-decay1e-4 batch512-1024 clip1.0 dropout0.12 temp0.08 infoNCE hard-neg0.2 same-sector feature-dropout0.12

## Gates v2 — IC>0.01 FAIL → No v2 Honest

From assets/eval_forward.json 2026-08-12T21:47Z n=233 trades_final_ranked_v6.csv:

| Metric | Value | Target | Verdict |
|---|---|---|---|
| IC 1M rank | 0.0051 n=233 | >0 | PASS weak |
| IC 3M | 0.0064 | >0 | PASS |
| IC 6M rankdata | 0.007 / spearman scipy 0.0097 | >0.01 for v2 | FAIL → no v2 |
| IC 12M | 0.0062 | >0 | PASS |
| Top50 IC | 0.079 | >0.01 | PASS small-n n=50<233 |
| calibration bias isotonic | before 5.76% (11.37→5.61) after 0.0% 162 thresholds | <1% PASS |
| isotonic IC preserve | 0.878→0.881 | — | PASS |
| triple-barrier +10% before -7% 63d | 0.2189 random0.25 | >0.25 | FAIL false miscalibrated |
| distress early-warning corr pred_fwd6 vs DD | -0.2624 higher pred→more distress proxy | >0 | FAIL inverted needs -corr*λ |
| entry_mean | 0.8409 thr0.7 | — | PASS |
| Sharpe mean0.0504 std0.1251 slip $0.01 | sqrtN 6.15 PASS >1.0 / sqrt2 ann6M 0.57 FAIL <1.2 | >1.2 | SPLIT ambiguous honest |
| n_trades | 233 | ≥200 Kelly needs | PASS min |

Gate per HILLCLIMB_v6_to_v2_REPORT.md: v2 promotion requires IC>0.01 AND Sharpe>1.2 sqrtN AND CQS>0.72 + market>0.58 + next_R2>0.20 — current IC 0.007/0.0097<0.01 FAIL, Sharpe 0.57/6.15 ambiguous FAIL 0DTE 1.2 stretch, CQS 0.7017<0.72 FAIL → no v2 promotion honest no fake promotion per spec.

### 0DTE Locked — ONLY IF IC>0.03 && Sharpe>1.2 && win>55% && DD<12% kill-switch separate bankroll NOT financial advice
- Tiny gated tiny spreads only never leverage user funds not financial advice
- Current IC 0.007<0.03 → no 0DTE live stays paper free platform forever Knowledge→Edge→Money
- Chain: Free platform free users → Kalshi 0.25 Kelly 1% max per play 3 concurrent max p vs market ≥5% edge → equity paper prop_edge_equities.jsonl validates forward → 0DTE tiny gated ONLY IF all gates paper first private edge separate bankroll kill-switch DD>12% or Sharpe<1.0 close all
- 0DTE details: IC>0.03 rankdata Spearman full 233 not Top50, Sharpe>1.2 sqrtN after $0.01 slippage or sqrt2 ann>1.0 stretch preferred 1.2, win>55% triple-barrier hit rate currently 21.89% FAIL → calibrate barrier horizon 63d→21d threshold +10%/-7%→+7%/-5%, DD<12% kill-switch separate bankroll, calibration bias 0.0 isotonic 162 thresholds preserves IC
- Bias check: distress -0.2624 inverted → loss += -corr*λ or invert head, isotonic bias 0.0 solved 5.76%→0.0, Top-50 0.079 would pass but n=50 too small for Kelly (needs 233 min)

### Construct Validity First (locked 2026-08-08)
- Define construct plain-English: forward = true_fwd 12M realized from for_history.json vs pred_fwd6 scaled 6M head rank IC Spearman measures future business return rank coherence not price momentum
- Operationalize actually measures: rank IC Spearman + calibration bias isotonic + triple-barrier hit + distress early-warning corr + entry_mean thr
- Convergent/discriminant/predictive: IC>0 proves embedding knows future business return not just label recall/purity, Top50 0.079 concentration shows conviction but small-n fails Kelly, distress inverted -0.2624 discriminant fail need invert loss, triple 0.2189<0.25 miscalibrated convergent fail
- Threats: lookahead FY leakage prevented by honest_split ticker 70/15/15 same-ticker next-FY recall, market-wide features same FY OK regime conditioning not future leakage YoY prior FY 5Y rolling z-score, small-n Top50 overfit threat
- No vanity metric: CQS composite 0.5*recall+0.5*purity+sector+market+continuity not recall-only fixes early restore bug, MAE/RMSE/R2 5-fold CV SHAP logged eval JSON glass-box Lab page

## Smoke vs Full Evidence

### Transformer 4,000 rows 2ep CPU — CQS0.5908 PASS lit
- CQS 0.5908 comp0.842 recall@10 0.9125 train fast val_recall0.926 purity0.7589 PASS >0.60 sector_acc0.13025 FAIL 11-way needs >2ep sector head starves CLS attention market_acc0.59325 PASS >0.58 next_R2 -0.0031 FAIL vs 0.20
- fusion transformer d64 d_model96 4L 4H tower_width24 tower_blocks1 drop0.12 temp0.08 hard-neg0.2 batch512 -u unbuffered fix buffered SIGTERM 74-88s no logs
- Signal: recall super fast 0.91→1.0 in 2ep proves transformer generalizes better early vs gated 0.4697 test recall 0.0 overfit small-data

### Gated 14.4k 2ep batch1024 crash
- report 0.0611 recall0.0 restored best 0.7017 remains honest
- Log /tmp/equities_trans_14400_smoke.log SIGTERM 74-88s buffered fixed python -u

### Full transformer 14.4k 60ep epoch0 (SIGTERM 167s)
- epoch0 loss6.0163 val_recall0.9 test_recall0.95 purity0.718 comp0.809 would beat 0.7017 if finished (comp0.809>0.7017) SIGTERM killed 167s before epoch1 Pty false timedOut false TRUNCATED
- Evidence: comp0.809>0.7017 would beat + expected CQS>0.72 if epoch1-60 completed, memory 14,400×154 ~6.8MB fine VM 7.8G, OOMGuard

## OODA 4/4 — MoMA-lite Router 5 Tiers

**Router:** router-pack MoMA-lite 5 tiers deterministic/llm/deep_research/action_operator/agentic_epic + GARNet 9 lenses history-penalized

### Observe — L0+L1
- Inputs: pipeline/data/mtnn_report.json assets/eval_forward.json eval_scoreboard.json train_matrix_v6_20f.npz
- Nodes: scout-prime-coord + strategist-bloom-forms (gate analysis Forms+Bloom) + strategist-acne-zep (tower auto-detect ACNE17n27e Zep CLS-RoPE) + strategist-kalm-vicreg (KaLM SHAP construct validity VICReg CORAL GRL SupCon)
- Output: orient_context.json

### Orient — L1→L2
- Lenses 9: Forms, Bloom, ACNE-17n27e, Zep, CLS-RoPE, VICReg, CORAL, GRL-SupCon, KaLM-SHAP
- Nodes: planner-dag7 + researcher-triage + deep-researcher-wide sweep 5-7 deep-research-pack
- Output: dag_spec.json + side_effect_tags READ/WRITE_IDEMPOTENT

### Decide — L2→L3
- Nodes: synthesist-weaver
- Output: weaver_plan.md ranked tasks 5 lanes max3/4

### Act — L3→L4
- Nodes: builder-maker + executor-elite OODA inner + operator-tempo :01 + action-operator-closer tool-first
- Output: artifacts + model.pt + eval.json triple-write

## L0 Scout-Prime Pacing :01 max3/4

- Layer 0 coordinator Ultra host OODA host
- Pacing: max3 concurrent max4 total pacing-filtered max3/4 swarm faster hillclimb_backoff all-lanes-busy-guard.js 1653B tempo :05 swarm faster conf0.82 applied 2026-08-12, ScoutCommsBus relevantAgents HandoffEnvelope 7 req, communication-pacing.js PacingFilter max3/4 tempo :13, tempo :01 ultra 3 LOCAL-GPU exempt <7 max
- Checkpoint: checkpoint-manager.js LangGraph pause/resume timeline.jsonl workspace/.scout/missions/<id>/timeline.jsonl mandatory 7-field nodeId agentId attempt latency tokens status errorClass ts runId ooda tempo :01 zero_deps true even no-change per checkpoint-manager
- Triple-verify 7/7 per lane2 verification bundles/ultra/runs + .scout/missions/_cron + .scout/missions/<id>
- Stuck-detector.js loop>3 conf<0.4 latency>thr → 1 honesty lens lateral-thinking-pack 9 lenses + honesty mechanics
- Honest signals: 503/unavailable never faked EXTRACTED vs INFERRED tagged no fabrication
- MissionLog writer bundles/scripts/mission_log.py

## L1 Strategist×3 9 Lenses

- 3 strategists history-penalized MoMA-lite router 5 tiers
- strategist-forms-bloom: Bloom depth construct validity + forward IC gate + Sharpe sqrtN vs sqrt2 + Forms taxonomy → gate_analysis.md forms_bloom.json
- strategist-acne-zep: ACNE 17n27e local-first 54 contacts 27 edge types graphify_constructs() stage4 optional no pip no cloud no vector DB no OAuth + Zep temporal FY12 FY emb 12-d regime + CLS-RoPE tower tokens + FY token + [CLS] 4L 4H d_model96 RoPE pos
- strategist-kalm-vicreg-coral-grl: KaLM SHAP model-agnostic glass-box permutation/Kernel SHAP 5-fold CV MAE/RMSE/R2 construct validity first no vanity + VICReg variance loss 0.2 push std 0.7%→4% invariance covariance + CORAL correlation alignment sector_acc 0.957 vs market 0.57 + GRL gradient reversal distress -0.2624 invert loss -corr*λ + SupCon temp0.08 infoNCE same-ticker adjacent FY + hard-neg boost 0.2 same-sector + feature-dropout0.12
- All :01 lite T5 max3/4 OODA Orient MoMA-lite router zero_deps true

## L2 Planner DAG7 Side-Effect Tagged

- Agent planner layer 2 DAG dynamic-planner
- Nodes 8 DAG edges: n0-fetch-external(EXTERNAL) → n1-build-v6-towers(WRITE_IDEMPOTENT) → n2-train-smoke-2ep(READ) → n3-train-full-60ep(WRITE) → n4-eval-forward(READ) + n5-eval-cqs(READ) → n6-money-gate(READ) → n7-promote-v2(WRITE_IDEMPOTENT)
- Tags: EXTERNAL GDELT yfinance GPR EPU GSCPI xls optional proxy fallback, READ data read-only no mutation eval_forward.json train_matrix.npz, WRITE model weights update full 60ep, WRITE_IDEMPOTENT train_matrix_v6_20f.npz pipeline/data/mtnn_best.pt cache JSON
- Input ask: build_real_v6_towers.py merges Z 14,400×118→154 offline fallback synthetic sector_context+form proxy + sector noise honest saves run; Memory: One canonical package dottie/rl thin re-export no sys.modules swap; One PM per app npm only package-lock.json Vercel
- FailureTaxonomy5: INPUT_CORRUPTION train_matrix_v5 missing→synthetic fallback honest 17→20 noise proxy saves run, CONTEXT_STARVATION SIGTERM 74s buffered no logs→fix -u unbuffered batch1024 lite + nano fallback, TOOL_FAILURE GDELT/yfinance timeout offline zero-deps true→proxy fallback sector noise + market-wide EPU/GPR static, REASONING_COLLAPSE sector_acc0.13 recall0.91 2ep too short sector starves CLS attention >2ep needed 60ep fixes, OUTPUT_CORRUPTION v2 fake promotion if IC<0.01 must refuse honest gate verifier kills fake

## L3 Swarm Concurrency 4 Pacing-Filtered :01

- 4 lanes max3/4 hillclimb_backoff :05 swarm faster conf0.82, :01 ultra 3 LOCAL-GPU exempt <7 max 3 concurrent exempt timeout counts 60ep 30-60m
- T1 lit e: researcher literature-echo fast triage lit survey 5-7 sources transformer CLS-RoPE vs gated MoMA-lite + VICReg SupCon ranking loss → bundles/research/hill133_t1_lit.md
- T2 triage: researcher fast-triage 2ep smoke 14.4k batch512 -u unbuffered python3 -u pipeline/train_mtnn.py --epochs 2 --dim64 --fusion transformer --batch512 --device auto gate CQS0.5908→0.72 path recall>0.9 purity>0.6 sector>0.95 → t2_triage.json :01 lite
- T3 deep-research: deep-researcher wide-sweep 5-7 deep-research-pack Observe/Orient heavy industry_event 10f GDELT 8-K + political_risk 10f GPR EPU WGI + trade_commodity 12f CL BZ HG SLX DXY GSCPI quantification YoY 5Y rolling z-score → bundles/research/hill133_t3_deep.md
- T4 tool: action-operator tool-first complex-actions-pack Act +9 prod practices build towers_v6/ industry_gdelt.py political.py trade_commodities.py + fetch_external.py cache 30d → pipeline/towers_v6/*.py pipeline/data/external_cache/ sideEffect WRITE_IDEMPOTENT
- T5 epic lite money: executor elite OODA inner :01 ultra 3 LOCAL-GPU exempt full 14.4k 60ep transformer CLS 4L 4H d_model96 OneCycle 10% warmup 60ep RTX4090 python3 -u pipeline/train_mtnn.py --epochs60 --dim64 --fusion transformer --batch512 --tower-width24 --d-model96 --n-fusion-layers4 --n-attn-heads4 --tower-blocks2 --mlp-heads --lr1.5e-3 --weight-decay1e-4 --val-every5 --device cuda --one-cycle --pct-start0.1 --clip1.0 --dropout0.12 --temp0.08 --hard-neg0.2 expected comp0.809>0.7017 val_recall0.9 test0.95 purity0.718 CQS>0.72 quit 14,400×154 6.8MB 7.8G fine nano fallback batch256 accum2 if OOM → pipeline/data/mtnn_best.pt 514K → equities_v6_money_best.pt or v2 if gates pass
- Loss T5: MSE fwd + rank_w1.0 256 pairs margin0.02 thr true diff>5% + var_w0.2 (std_true-std_pred)^2*0.2 push 0.7%→4% + w_f6=5 + w_dd1.5 + w_vol0.3 + w_entry2.0 + w_nce1.0 weight-decay1e-4 dropout0.12 temp0.08 hard-neg0.2 feature-drop0.12
- LCG: dailySeed=YYYYMMDD UTC int e.g. 20260812 glibc LCG state=(seed*1103515245+12345)&0x7fffffff idx=state%N N=4831 equities 20719 unified idx3970 triple[3970,14390,4582] chimera A+B=C same-link-same-stars Python&Node agree play.html:680 build_chimera_from_towers.py:219 site-nav.js:5
- PWA: v67 CACHE_NAME vector-equities-v67-dark 74426B sw.js HIT void #080A0F cards #FFFEF7 ink #14181d Okabe archetype manifest theme_color #0b0e14 display standalone scope / shortcuts Play Daily+Lab fuse A+B→nearest real as served index.html
- Drag-map→Jordan: 3D embedding map central shared-map.js LOD4000/8000 DPR1 fillRect vintage drag inertia Jordan entry shows position not auto-start game Popular tap-to-explore Players Explorer points visible dark bg #080A0F single-select clear previous
- Torch: auto cuda if available else cpu CPU Hatch VM GPU Alienware LOCAL-GPU per MEMORY training split 2026-08-10 08:29 auto device nano batch fallback if OOM batch256 accum2

## L4 Verification Econ Budget3 thr8.0 earlyExit0.3 fix-once

- critic L4 QA 0-10 + forensic-auditor second brain verifier-with-budget.js v5 Prime single enforcement budget3 thr8.0 earlyExit0.3 fix once if <8 max2 loops total early_exit if delta<0.3 plateau
- Rules: ship if ≥8.0, fix once if <8, max2 loops, no_fake EXTRACTED vs INFERRED tagged 503 never faked honest gate, early_exit delta<0.3, max2 loops single enforcement point
- SHAP construct validity: model-agnostic explainer Kernel SHAP or permutation importance + partial dependence logged eval JSON glass-box Lab page CV 5-fold MAE/RMSE/R2 construct validity first define construct plain-English → operationalize actually measures → convergent/discriminant/predictive → threats no vanity metric tags MTNN MTL multi-tower CLS-RoPE SupCon VICReg/CORAL GRL distress invert ranking margin variance push
- CQS gate: 0.5*recall@10+0.5*purity@20+sector+market+continuity vs baseline 0.605 baseline_label_permutation0.1106 random0.1117 vs 0.7057 score lift6.32x target 0.7017→0.72+ (+0.0183) expected via transformer 60ep finishing
- Money gate 0DTE locked ONLY IF IC>0.03 AND Sharpe>1.2 AND win>55% AND DD<12% kill-switch separate bankroll NOT financial advice current IC0.007/0.0097<0.03 FAIL Sharpe sqrt2 0.57 FAIL Top50 0.079 small-n fail n=50<233 → no 0DTE stays paper Kalshi 0.25 Kelly1% max3 concurrent private paper equity paper prop_edge_equities.jsonl free platform free users 0.9 no Stripe
- Metrics hook: bundles/ultra/metrics_hook.js observability_tick */15 collects per-run OODA4/4 agentic6/6 tempo:01 MoMA5 tiers graph sizes checkpoint health verification scores pacing stats max3/4 → bundles/ultra/runs/metrics.jsonl
- Triple-write 7-field even no-change mandatory fields nodeId agentId attempt latency tokens status errorClass ts runId ooda tempo :01 zero_deps true per checkpoint-manager spec workspace/.scout/missions/_cron/timeline.jsonl + bundles/ultra/runs/t5-hill133/timeline.jsonl

## Failure Taxonomy 5 + Side-Effect 4 + Recovery Ladder retry→patch→replan→escalate

- INPUT_CORRUPTION → retry synthetic fallback offline proxy 17→20 noise honest saves run
- CONTEXT_STARVATION SIGTERM → patch -u unbuffered batch1024 lite + nano batch256 accum2 fallback OOMGuard
- TOOL_FAILURE GDELT/yfinance timeout → replan 20→17 fallback if 32 noisy w_f6=5 ranking loss 256 pairs margin0.02 thr diff>5% + var loss0.2 std push 0.7%→4% + distress invert -corr*λ + triple-barrier 63d→21d thr +10%/-7%→+7%/-5%
- REASONING_COLLAPSE sector_acc low → replan 60ep fixes gated→transformer token fusion CLS attention sector head >2ep needed
- OUTPUT_CORRUPTION v2 fake promotion → escalate verifier kills fake CQS>0.72 required honest gate
- Ladder: retry gated lite 14.4k 2ep batch1024 -u → patch transformer→gated if OOM batch256 accum2 fallback OneCycle10% warmup clip1.0 drop0.12 temp0.08 hard-neg0.2 → replan 20→17 towers fallback if 32 noisy w_f6=5 ranking 256 pairs margin0.02 var0.2 + distress invert + triple calibrate → escalate LOCAL-GPU RTX4090 60ep required COORDINATION_LOCAL_GPU.md claim COORDINATION.md + GitHub sync + Tasks + push Vercel auto-deploy market-review before publish scrub Hatch/Meta refs verify production-grade
- Stuck Detector lateral lens 9: loop>3 conf<0.4 latency>thr →1 honesty lens stuck-detector.js lateral-thinking-pack

## LOCAL-GPU Exempt 3 <7 Max

- 3 nodes exempt <7 max: n3-train-full-60ep T5 epic lite, n4-eval-forward IC Sharpe, n5-eval-cqs CQS
- Exempt rule :01 ultra 3 LOCAL-GPU exempt <7 max clear stale 2h hot T-learning 1m ultra 2026-08-12 sweep last20 fails + SIGTERM OOMGuard 167s comp0.809 14.4k 60ep + MAE0.2313 vs 0.2085 no-fake → guards updated v1.1 :01 ultra 3 LOCAL-GPU exempt <7 max clear stale 2h hot
- Cmd: python3 -u pipeline/train_mtnn.py --epochs60 --dim64 --fusion transformer --batch512 --tower-width24 --d-model96 --n-fusion-layers4 --n-attn-heads4 --tower-blocks2 --mlp-heads --lr1.5e-3 --weight-decay1e-4 --val-every5 --device cuda --one-cycle --pct-start0.1 --clip1.0 --dropout0.12 --temp0.08 --hard-neg0.2 — nano fallback --batch256 --accum2 if OOM
- Device auto torch auto cuda if available else cpu CPU Hatch VM GPU Alienware LOCAL-GPU per MEMORY training split 2026-08-10 08:29 auto device 90s poll morning/evening briefs extract todos sync GitHub+Tasks hook_podcast_brief_auto_exec__interval@90s.json
- One canonical runs bundles/ultra/runs/ only prune100 max monthly dupe mirrors apps/dottie/pipeline/ gone lesson0.88
- One canonical package dottie/rl canonical ava/rl thin re-export no sys.modules replacement lesson0.92 re-export only never swap namespace kills submodule imports
- One PM per app apps/arxiviq npm only package-lock.json No dual bun.lock lesson0.85 Vercel

## Modeling Rule (locked 2026-08-08)

- Train real models ≥2 5-fold CV MAE/RMSE/R2 model-agnostic SHAP/permutation importance glass-box log + construct validity define construct plain-English operationalize check convergent/discriminant/predictive document threats no vanity metric
- Unified multi-tower multitask deep neural networks preferred endgame MTNN tower_blocks2 d_hidden96 fy_emb12 d_emb64 fusion ContinuousFusion gated + transformer CLS 4L 4H d_model96 d_tower24 attention over n_towers 20
- Applies to vector-hoops Front Office draft expectation cap efficiency foresight surplus win projections contract valuation any future model
- SHAP logged eval JSON glass-box Lab page

## Deliverables Checklist

- [x] CQS 0.7017 PASS baseline 0.605 +0.0967 514K 300K backbone recall@10 1.0 purity0.68 sector_acc0.9567 continuity0.72 FY12 14400×118→154
- [x] Forward IC 1M0.0051 3M0.0064 6M0.007/0.0097 spearman 12M0.0062 Top500.079 bias0.0 isotonic 162 thresholds triple0.2189 distress-0.2624 Sharpe sqrt2 0.57 FAIL sqrtN6.15 PASS → IC>0.01 FAIL gate no v2 honest
- [x] Smoke 2ep transformer 4k CQS0.5908 recall0.9125 purity0.7589 sector0.13 market0.593 PASS >0.58 next_R2 -0.0031 FAIL vs full 14.4k 60ep epoch0 loss6.0163 val_recall0.9 test0.95 purity0.718 comp0.809 would beat 0.7017 SIGTERM 167s before epoch1
- [x] Tower V6 17→20 10+10+12=32 feats industry_event/political_risk/global_trade_commodity pipeline towers_v6/ synthetic fallback offline auto-detect families model no code change ContinuousFusion gated learns VIX weighting
- [x] Free platform free users no Stripe free forever Knowledge→Edge→Money 0.25 Kelly1% max3 concurrent Kalshi → equity paper → tiny 0DTE gated ONLY IC>0.03 Sharpe>1.2 win>55% DD<12% kill-switch separate bankroll NOT financial advice
- [x] Zero-deps true torch auto cuda else cpu LOCAL-GPU resume batch nano OneCycle10% warmup clip1.0 LCG20260812 idx3970 PWA v67
- [x] DAG7 JSON PAL usable from Claude Code agents tag/provenance rogue protection MoMA-lite 5 tiers GARNet OODA4/4 L0 pacing max3/4 :01 L1×3 9 lenses L2 side-effect READ/WRITE_IDEMPOTENT L3 swarm concurrency4 T1 lite T2 triage T3 deep-research T4 tool T5 epic lite L4 budget3 thr8.0 earlyExit0.3 fix-once SHAP construct validity palindrome
- [x] This file docs/T5_hill133_planner.md

**No financial advice. Platform free forever. Profit via own calibrated edge private 0.25 Kelly1% max3 concurrent per-play separate bankroll kill-switch DD<12% — IC>0.03 gate.**

Lens pause — resume LOCAL-GPU 60ep transformer via COORDINATION_LOCAL_GPU.md claim python3 -u pipeline/train_mtnn.py --epochs60 --fusion transformer --batch512 --device cuda

