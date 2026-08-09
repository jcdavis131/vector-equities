# Vector Equities

![CI](https://github.com/jcdavis131/vector-equities/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

A daily equities "chimera" puzzle over SEC EDGAR 500 tickers embedding space — guess the blend of real company-FYs behind each composite. Static site, no backend, live at https://equities.dumbmodel.com.

> Solo personal project, no connection to employer, built with public/free-tier only (free data pipeline, static Vercel).

> **Picking up in-progress work?** Start at [`docs/HANDOFF.md`](docs/HANDOFF.md) — current state 2026-08-09: 4,831 FYs, 500 tickers, 64-d purity 0.7057, verification commands, and open follow-ups.

## The embedding

4,831 company-FYs across 500 tickers (2015–2024) from SEC EDGAR XBRL CompanyFacts + market + 10-K text chunks. Per-FY z-scored and winsorized ±4 so eras and sectors compare honestly.

A multi-tower neural net fuses 17 families into a 64-d L2-normalized vector:

- **17 towers:** income, balance, cashflow, growth, profitability, leverage/liquidity, efficiency, per-share, market_price, valuation, management_neo, ownership, disclosure_text, sector_context, macro_regime, form, bbref_bridge — each `cat([x·m, m]) → 96h → 24d` with LayerNorm + skip + L2. Conceptually covers classic factors: Altman Z-score, Piotroski F-score, Beneish M-score, Sloan accruals, Quality Minus Junk (QMJ), momentum, net insider flow, fog/readability, etc. Concrete columns listed in `docs/DATA_MODEL_equities.md` and `assets/feature_manifest_v6_real.json`.
- **Wiki-text tower:** 16-d reduced from 384-d MiniLM (optional 18th family, concatenated when present).
- **Fusion:** 17 tokens × 40-d → proj 128, plus CLS + FY embedding 12-d → 128 = 19 tokens, transformer `d_model 128, 4 layers, 4 heads, ff 512, pre-LN, dropout 0.15` → CLS `128→512→64 L2`. Positional scalar `/252` for era alignment gated by `year_norm_excluded_from_X` (prevents leakage).
- **Heads:** 8 archetypes, 11 GICS sectors, 14-d profile, next-year profile, 12 skill grades, valuation, market directional, vol, health, payout, mgmt quality, ownership concentration.

On the published geometry (`assets/real_data.json`, model `equities_mtnn_v_rebuild_d64_transformer`):

- **k-NN sector purity@10:** 0.7057 (baseline random 0.1117) — **lift 6.32×**, n=4,831 rows / 500 tickers / 11 sectors, 64-d. Gate >0.65 `tests/test_eval_sector_coherence.py` PASS.
- **cross-ticker purity@10** (same-ticker neighbors excluded to remove trivial same-ticker inflation from same-ticker adjacent-FY contrastive): 0.4013 (baseline 0.1117) — lift 3.59×, gate >0.35 PASS.
- **silhouette cosine:** -0.0034 vs label-permutation -0.0204 (range [-1,1], sector clusters overlap but separate above chance).
- **Forward gate:** IC rank 1m 0.0051 / 3m 0.0064 / 6m 0.007 / 12m 0.0062, triple-barrier 10% before -7% 63d 0.2189, n_trades 233 — gate IC>0 proves embedding knows business future not just label.
- **Composite** (see `assets/eval_scoreboard.json`): 0.5*sector gate + 0.3*cross + 0.2*forward IC>0 indicator + silhouette bonus — used for promotion tracking like hoops CQS.

Regenerate metrics with `python pipeline/eval_sector_coherence.py` → `assets/eval_sector_coherence.json` and `python pipeline/composite_score.py` → scoreboard. Eval files committed for static-host auditability.

## The site

Plain HTML/JS/Canvas, no framework, PWA-capable (`sw.js`, `offline.html`). Pages: the daily game (Guess The Ticker 6 tries, 64-d cosine, sector/arch clues, same for everyone UTC), 3D embedding map (PCA → xyz), company dossiers (full FY trajectory), trends (sector momentum), model lab (A+B → nearest real fusion), methods.

Static assets:

- `assets/real_data.json` — points with xyz, 12 skill grades, embeddings, FY chain
- `assets/real_pca.json` / `real_pca_full.json` — precomputed PCA for sky
- `assets/manifest.json` — PWA manifest (also at `/manifest.json` for no-404 parity)
- `assets/eval_sector_coherence.json` + `assets/eval_scoreboard.json` — shipped eval

The site runs from Vercel static — no server, client-side cosine.

## Data pipeline

```bash
python pipeline/fetch_sec_summary.py --limit 300          # SEC EDGAR XBRL CompanyFacts + sub index -> pipeline/data/chunks summary
python pipeline/build_real_from_summary.py --limit 300    # 138 feats 17 families + wiki 16-d, per-FY z-score winsor ±4
python pipeline/build_skills.py                            # 12 skills: Profitability Growth Moat CashConv CapAlloc BalHealth Efficiency ValDisc Mom MgmtQuality Yield Disclosure
python pipeline/build_archetypes.py                        # 8 archetypes k-means on financial profile: Compounder Cash_Cow Turnaround HyperGrowth Heavy_Industrial Bank_Cap Heavy Moonshot_Bio Serial_Acquirer
python pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128   # 4L 4H transformer fusion MTNN v6 towers real
python pipeline/regen_assets.py            # or pipeline/export_real_assets.py -> assets/real_data.json + pca + manifest copy
python pipeline/eval_sector_coherence.py   # -> assets/eval_sector_coherence.json 0.7057 / 0.4013
python pipeline/eval_forward.py            # -> forward IC gate IC>0
```

Sources: SEC EDGAR (free, User-Agent + throttle), yfinance market, DEF14A NEO parser. Every response cached under `pipeline/cache/` and `pipeline/data/`; reruns resume, `--offline` rebuilds from cache only. Rebuilds gated by `tests/test_no_ticker_leakage.py` (no ticker in feature spec, FY embedding 12-d excluded from tower X, coverage scalar prevents zero-impute bias, year_norm excluded from X pos-proj) and `tests/test_eval_sector_coherence.py` (>0.65 / >0.35).

## Training

`pipeline/train_mtnn.py` (torch) trains Equities MTNN — 17× ResidualTower, gated/concat/transformer fusion, FY embedding 12-d, 64-d L2 norm, multi-head: archetype 8, sector 11, profile 14, next_profile 14, 12 skill towers, valuation, market, vol, health, payout, mgmt, own, plus InfoNCE same-ticker adjacent-FY contrastive with sector hard negatives 0.3 boost.

- **Optimizer:** AdamW no-decay biases, OneCycle 10% warmup linear, grad clip
- **Losses:** masked MSE for profile heads, cross-entropy archetype/sector, SupCon hybrid sector/quality optional, CORAL/GRL regulator optional for era invariance
- **Promotion gated:** sector purity >0.65 + cross-ticker >0.35 + IC>0 (see `assets/eval_scoreboard.json` `metrics.knn_sector_purity_at_10.threshold_gate` 0.65, `cross_ticker` 0.35, `forward.gate_ic_gt_zero`). Transparent 64-d contract stays until candidate beats it on leak-free eval — no season leak, FY-level split, no ticker feature.
- **Research notes:** `docs/TOWER_V6_DESIGN.md`, `docs/ARCHITECTURE.md`, `docs/REAL_PROD_REPORT.md`, `docs/PLAN.md`. Candidate gates `candidate.json` first, promote only if beats shipped.

Optional v6 real towers: `pipeline/build_real_v6_towers_real.py` → 17 families + wiki tower, `pipeline/train_career_mtnn_v6.py`. Local GPU lane for heavy epochs (runner OOM, torch wheel).

## Running locally

```bash
python -m http.server 8000   # static site, open http://localhost:8000
python -m pytest -q          # pipeline gates (needs dev extras in pyproject.toml) — sector coherence + no-ticker leakage + no-fabricated embedding + calibration provenance
```

## License

MIT. Solo personal project, no connection to employer, built with public/free-tier only.

