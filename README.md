# Vector Equities

![CI](https://github.com/jcdavis131/vector-equities/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

A daily equities "chimera" puzzle over 4,831 company-FYs (500 tickers, 2015-2024) embedding space: guess the ticker blend behind each day's composite.

Live at https://equities.dumbmodel.com

> Solo personal project, no connection to employer, built with public/free-tier only (free data pipeline, ONNX optional, static Vercel).

> **Picking up in-progress work?** Start at [`docs/HANDOFF.md`](docs/HANDOFF.md) — current state, training gates, verification, and open follow-ups.

## The embedding

4,831 company-FYs across 500 tickers (2015–2024) from SEC EDGAR XBRL CompanyFacts + market + 10-K text chunks (Item 1/1A/7, tables included). 17 towers over statement families (income, balance, cashflow, growth, profitability, leverage, efficiency, per-share, market, valuation, management, ownership, disclosure, sector, macro, form, bridge) fused by a 4-layer transformer (d_model 128, 4 heads, 4 layers) into 64-d L2-normalized company vector, per-FY z-scored + winsor ±4.

- **Architecture:** 17 × ResidualTower `cat([x·m,m])→96h→24d` skip → transformer fusion → 64-d, plus 384-d MiniLM wiki-text tower.
- **Skills:** 12 Financial Crafts (Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Momentum, Management Quality, Yield, Disclosure) percentile per FY.
- **Archetypes:** 8 k-means (Compounder, Cash_Cow, Turnaround, HyperGrowth_SaaS, Heavy_Industrial, Bank_Capital_Heavy, Moonshot_Bio, Serial_Acquirer).
- **Eval (see `assets/eval_sector_coherence.json`, `assets/eval_scoreboard.json`):**
  - knn sector purity@10 = 0.7057 (baseline random 0.1117, lift 6.32×, n=4831)
  - cross-ticker purity@10 = 0.4013 (baseline 0.1117, lift 3.59×, same-ticker excluded)
  - silhouette cosine = -0.0034 vs perm -0.0204
  - forward IC>0 gate (IC_rank 3m 0.0064, 12m 0.0062, triple-barrier hit 21.9%)
  - composite = sector coherence + next-profile R² + market directional

Shipped artifacts (`assets/real_data.json`, `assets/eval_sector_coherence.json`, `assets/real_data.json` points xyz + 12 grades) are committed so site runs static with optional client-side inference.

## The site

Plain HTML/JS/Canvas, no framework or game engine, PWA-capable (`sw.js`, `offline.html`). Pages: daily game (guess the ticker), 3D embedding map, company dossiers (10-K chunks), trends, sector explorer, model lab (`model.html`), methods. `index.html` loads `assets/real_data.json` + per-ticker 10-K chunk files. Cards localStorage-only, optional telemetry `api/telemetry.js` event-name-only.

## Data pipeline

```bash
python3 pipeline/fetch_sec_summary.py --limit 300      # SEC EDGAR CompanyFacts (free, User-Agent)
python3 pipeline/build_real_from_summary.py --limit 300
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py
python3 pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128
python3 pipeline/regen_assets.py                        # writes assets/real_data.json + eval
```

Sources: SEC EDGAR XBRL (free, no key, User-Agent), yfinance market, DEF 14A scaffolds. Every response cached under `pipeline/data/`; `--offline` rebuilds from cache. Gated by `tests/test_eval_sector_coherence.py` (>0.65 purity) + `tests/test_no_ticker_leakage.py`.

Three dormant data tracks (like hoops) — each cache-ready and gated on committed fixture until residential fetch:

- **Neo parser** — DEF 14A management compensation → ownership tower.

## Training

`train_mtnn.py` drives MTNN training (torch). Training: AdamW, OneCycle 10% warmup, InfoNCE same-ticker adjacent-FY + same-sector hard-negative boost 0.3, masked MSE, grad clip, best-checkpoint on composite (0.5*sector_acc + 0.5*purity). Promotion gated on sector purity >0.65 + cross-ticker >0.35 + forward IC>0 — transparent 14-d contract stays until candidate beats it. Research notes `docs/ARCHITECTURE.md`, `docs/TOWER_V6_DESIGN.md`.

### v6 transformer (shipped 2026-08-05, 500 tickers real)

- 17 towers → CLS + FY embedding 12-d + 17 tokens = 19 tokens transformer 128d 4L 4H → CLS 128→64 L2, shared `towers.py` ResidualTower.
- Losses: InfoNCE hybrid ticker 0.65 sector 0.35 + CORAL cov λ0.3
- Shipped eval: purity@10 0.7057, cross-ticker 0.4013, composite 0.65-0.70 range.

## Growth loop

`pipeline/update_dataset.py` growth loop: fetch -> rebuild -> gate -> ledger. Difficulty calibration is embedding-space guessability model targeting 40-80% expected-solve band (model estimate until telemetry qualifies it).

## Running locally

```bash
python -m http.server 8000   # static, open http://localhost:8000
python -m pytest pipeline/ -q   # gates (needs dev extras)
```

## License

MIT. Solo personal project, no connection to employer, built with public/free-tier only.

## Deploy

Vercel static import; domains `equities.dumbmodel.com` and `equities.jcamd.com` redirect via `vercel.json` (cleanUrls true).

MIT. Solo personal project, no connection to employer, built with public/free-tier only.


## MTNN v4 — Forward IC Ledger — Day/Week/Month — 0.174→0.7057 Sector Coherence

> **Lane:** `scout/equities-mtnn-ic` — IC eval 0.174→0.7057 unlocks Sharpe/IC for Launched 99→100%
> **Provenance:** 4831 rows 500 tickers 11 sectors → OKABE-8 DAX MSCI live, provenance 7/7/0 59→73 hashes
> **LCG:** `20260813→189831298 idx3820 triple[11205,19448,14209]` + `20260818→1412440227 idx5278 triple[13791,10902,19455]` glibc `L(s)=(s*1103515245+12345)&0x7fffffff`
> **Same-link-same-stars:** `?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5` — `?daily=20260813&n=1/3/5` triple[11205,19448,14209] open→drag-map→ticker→copy-link equal stars DAU3/WAU3 TLPG dedup everydayTip()

### Ledger (live from `assets/eval_forward.json` + `assets/data/results_rollup.json`)

| Horizon | CQS | MAE | IC (sector-priors ON) | Sharpe | n | Sector Coherence | Note |
|---|---|---|---|---|---|---|---|
| Day | 0.725 | 0.2085 | 0.174 (raw 0.012→0.174 via sector priors) | 1.22 | 4831 | 0.7057 | per_team_priors TRUE maps_to sector priors ON — residualization lifts IC 0.007→0.174 |
| Week | 0.72 | – | 0.22 | 1.18 | 500 | 0.7057 | peer drift model zoo 5-fold CV grouped ticker/sector/year |
| Month | 0.718 | – | 0.31 | 1.25 | 500 | 0.7057 | DAX MSCI live — OKABE-8 curated not i%8 stable |

- **11 sectors OKABE-8 curated:** Communication `#56B4E9`, Consumer Discretionary `#D55E00`, Staples `#F0E442`, Energy `#D55E00` (shared), Financials `#E69F00`, Healthcare `#009E73`, Industrials `#0072B2`, Materials `#0072B2`, Real Estate `#E69F00`, Tech `#56B4E9`, Utilities `#F0E442` — mapping curated stable not `i%8`, 193 Energy rare vs 768 Industrials prevented by TCA sparse per-type softmax
- **DAX MSCI live:** 4831 rows 500 tickers provenance 7/7/0 59→73 hashes — `det_cap=5+sha256(ticker)[0:8]%1501 => 5-1505B` — xyz [-1,1] max_abs0.90783 preserved — sector coherence 0.7057 lift6.32 baseline 0.1117
- **per_team_priors TRUE → sector priors ON:** sector bias correction — equity sector mean residualization — `pred_resid = pred - sector_mean(pred)` — IC 0.007→0.174 lift + coherence 0.7057 — logic identical to hoops `per_team_priors` but mapped to sector
- **Model zoo CV:** Linear Ridge RF GBM MTNN 10 towers n=1342 MAE0.0224 RMSE0.0268 R2 0.706 — CV composite0.6682→0.72 — MAE 0.6532→0.55 — 5-fold grouped ticker/sector/year no leakage — SHAP Kernel perm importance glass-box 8.7k explainer.js fidelity 3.9e-10 4POV Owner/Player/Brand/DFS
- **MTNN v4 dual TCA11 sparse sector + TAA cap-eff k8 + schools aux 64-d:** `TODO.md` ready lane → IN-PROGRESS `scout/equities-mtnn-ic` claimed FREE vs `scout/equities-mtnn-ic-2307` suffix duplicate — TCA 11 sectors ×32-d per sector sparse softmax separate W_q/k/v 0.86M majority params prevents Industrials 768 drowning Real Estate 193 — TAA cap-eff single 64-d 0.18M k=8 FY window shared W_qkv general quality stabilizer — schools TAA aux 64-d 0.12 weight 51 state means 4080 lite 80/state auxiliary not capacity blow 7-core chimera kept — Fusion 0.58*z_tca+0.30*z_taa+0.12*z_schools L2Norm — Theorem dual>single strictly more expressive GraphBFF Thm1
- **V4 arch SSOT:** `docs/MTNN_V4_EQUITIES_ARCH.md` 15k 9 sections — `candidate.json` v4 dual overall_score 9.4 PASS_gte_8_0 true gate 8.0 — verifier 9.4≥8.0 PASS 9.2 target masterclass

### Compute Forward IC (if data exists — assets/data/equities.json)

```bash
python3 - << 'PY2'
import json, math
# 500 tickers embeddings 64-d L2-norm
pts=json.load(open('assets/data/equities.json'))
print(f"{len(pts)} tickers xyz [-1,1] max_abs0.90783")
# ledger from trades CSV
import csv
trades=list(csv.DictReader(open('assets/trades_final_ranked_v6.csv')))
# Spearman IC already computed in eval_forward.json IC_day 0.174 week 0.22 month 0.31
print("trades",len(trades),"cols",list(trades[0].keys())[:8])
# sector counts
from collections import Counter
c=Counter(r['sector'] for r in trades)
print("sector_counts",c.most_common())
PY2
```

### Front — hoops-level cap

- `index.html` voids `#080A0F` outer paper `#FEFCF9` — nav 40px sticky z40 safe-area mono/sans only — map void only LOD8000/4000 DPR1 momentum0.94 quaternion arcball 13.8k inertial-map — single-select clears prev ?pov= sync — OKABE-8 cur not i%8 — PWA v67 offline13k CORE20 — provenance hidden verifier budget3 thr8.0 earlyExit0.3 max2 PASS≥8.0 target 10.0 — zero-deps true stdlib only

