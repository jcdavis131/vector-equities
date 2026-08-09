# Vector Equities — Data Model & Storage (2026-08-09)

Solo personal project, no connection to employer, built with public/free-tier only.

## Dataset snapshot (served 2026-08-05)

- `n_rows_total` 4,831 company-FYs, `n_companies` 500 tickers, `n_sectors` 11 GICS — see `assets/eval_sector_coherence.json`
- FY range 2015-2024 sparse (some tickers IPO later)
- Embedding `equities_mtnn_v_rebuild_d64_transformer` dim 64 L2-normalized, served by `assets/real_data.json`
- Purity@10 0.7057 baseline 0.1117 lift 6.32×, cross-ticker 0.4013 lift 3.59×, silhouette -0.0034 vs -0.0204 permutation — see `assets/eval_scoreboard.json`

## Feature families (17 towers + wiki optional)

Concrete list from `assets/feature_manifest_v6_real.json` — 138 features (122 + 16 wiki) in:

| tower | n | cols (conceptual coverage) |
|---|---|---|
| income | 15 | REV COGS GROSS_PROFIT OP_INCOME EBITDA NET_INCOME EBIT GROSS_MARGIN OP_MARGIN NET_MARGIN EBITDA_MARGIN etc |
| balance | 10 | TOTAL_ASSETS TOTAL_LIABILITIES EQUITY CASH DEBT BOOK_VALUE TANGIBLE_BOOK WORKING_CAPITAL NET_DEBT INVESTED_CAPITAL |
| cashflow | 7 | OCF CAPEX FCF FCF_MARGIN OCF_TO_NET FCF_CONVERSION CAPEX_TO_REV |
| growth | 9 | REV_YOY EBITDA_YOY NET_YOY FCF_YOY REV_3Y_CAGR EBITDA_3Y_CAGR EPS_3Y_CAGR BOOK_3Y_CAGR OCF_3Y_CAGR |
| profitability | 5 | ROE ROA ROIC PROF_GROSS_MARGIN etc + FCF_ROIC ROIC_WACC_SPREAD |
| leverage_liquidity | 7 | CURRENT_RATIO QUICK_RATIO DEBT_TO_EQUITY DEBT_TO_EBITDA INTEREST_COVERAGE DEBT_TO_ASSETS NET_DEBT_TO_EBITDA |
| efficiency | 5 | ASSET_TURNOVER INVENTORY_TURNOVER RECEIVABLE_TURNOVER CASH_CONVERSION_CYCLE CAPEX_TO_DEPRE |
| per_share | 5 | EPS_DILUTED BVPS FCF_P S SHARES_YOY DILUTION_3Y |
| market_price | 10 | RET_1M RET_3M RET_6M RET_12M VOL_30D VOL_90D VOL_252D BETA_1Y VOLUME_AVG_30D MOMENTUM_12_1 |
| valuation | 8 | PE PB PS EV_EBITDA EV_SALES EARNINGS_YIELD FCF_YIELD DIV_YIELD |
| management_neo | 14 | NEO_COUNT CEO_AGE CEO_TENURE CEO_FOUNDER_FLAG CEO_TOTAL_COMP CEO_EQUITY_PCT AVG_NEO_COMP CEO_PAY_RATIO BOARD_INDEP_PCT BOARD_SIZE INSIDER_OWN_PCT CEO_PAY_VS_SECTOR NEO_TURNOVER CEO_DUALITY |
| ownership | 6 | INST_PCT INST_DELTA_QOQ INSIDER_NET_12M FLOAT_PCT TOP10_INST_CONC SHORT_INTEREST_PCT |
| disclosure_text | 6 | MDA_LENGTH MDA_SENTIMENT RISK_FACTOR_COUNT RISK_CHANGE_YOY FOG_INDEX_PROXY TONE_UNCERTAINTY |
| sector_context | 3 | SECTOR_REL_RET_12M SECTOR_CONCENTRATION SECTOR_BETA |
| macro_regime | 4 | RATE_10Y VIX_AVG_FY CREDIT_SPREAD_PROXY GDP_GROWTH_FY |
| form | 6 | EARN_SURPRISE_STREAK GUIDANCE_RAISE_FLAG EPS_REVISION_UP_PCT PRICE_VS_52W_HIGH ... (quality signals) |
| bbref_bridge | 2 | placeholders for cross-sport parity |
| wiki_embeddings | 16 | reduced from 384-d MiniLM — optional 18th family |

**Conceptual factor mapping:** Altman Z (working capital / leverage), Piotroski F (profitability/growth/liquidity), Beneish M (accruals / margin spikes), Sloan accruals (OCF_TO_NET, FCF_CONVERSION), QMJ (ROE/ROA/ROIC + growth + safety), etc — implemented as handcrafted ratios above, not vendor data.

## Normalization

- Per-FY z-score: μ/σ computed on that fiscal year cohort only, so eras compare honestly (like hoops per-season z-score for era fairness). Prevents lookahead across years.
- Winsorize ±4 σ after z — clips outliers beyond 4 std (default 4, not 3, to keep heavy tails in small caps but avoid NaN drift). Implemented in `pipeline/build_real_from_summary.py` and `pipeline/parse_neo.py`.
- Mask scalar `m` passed as `[x·m, m]` per tower input (tower sees 0-filled for missing + binary presence). Mean mask coverage scalar prevents zero-impute bias — gated by `tests/test_no_ticker_leakage.py` `coverage_scalar_mean_mask` true.
- FY embedding 12-d excluded from tower X (`fy_embedding_12d_excluded_from_tower_X` true) — only fused as positional token, not input feature, to prevent leakage `year_norm_excluded_from_X_pos_proj_gated` true.
- Ticker symbol never in feature spec (`no_ticker_in_feature_spec` true) — validated.

## Skills (12)

From `pipeline/build_skills.py` — percentile per FY within that year:

1. Profitability — ROE/ROA/ROIC blended margin
2. Growth — REV/EBITDA/EPS 3Y CAGR + YoY
3. Moat — gross + op margin stability, ROIC spread
4. Cash Conversion — OCF_TO_NET, FCF_CONVERSION, CAPEX_TO_REV
5. Capital Allocation — buyback vs dilution, ROIC vs WACC
6. Balance Health — current/quick, debt/equity/EBITDA, interest coverage
7. Efficiency — asset/inventory/receivable turns, CCC
8. Valuation Discipline — earnings/FCF yield, PE/PB/PS tilt (inverse)
9. Momentum — RET_12M-1M, 12_1 momentum label only, not trained as target except gate
10. Management Quality — CEO tenure, founder flag, pay ratio, board indep, duality
11. Yield — div + buyback yield, FCF yield
12. Disclosure Quality — MDA length / fog inverse, sentiment, risk change

Grades 0-99 transparent fallback when MTNN heads absent; client-side probe weights in `skills.json` equivalent `skill_probe` logic for PC labels.

## Archetypes (8)

K-means on 14-d financial profile (profitability, growth, leverage, efficiency, valuation, size) — `pipeline/build_archetypes.py`:

- Compounder — high ROIC + steady growth, moderate leverage
- Cash_Cow — high FCF margin, low growth, high yield
- Turnaround — negative momentum but improving OCF
- HyperGrowth_SaaS — high PS, high growth, negative earnings
- Heavy_Industrial — asset heavy, low turnover, high capex
- Bank_Capital_Heavy — leverage high, macro sensitive (Financials)
- Moonshot_Bio — cash burn, R&D placeholder, high vol (Health)
- Serial_Acquirer — high intangible growth, goodwill spike

Assignments in `assets/real_data.json` per FY, lite list in `assets/archetype_assignments.json` style.

## Embedding / Assets alignment — verified

- `assets/real_data.json`: 4,831 entries `{ticker, fy, sector, arch, skills[12], emb[64], x,y,z}` — critical core, used for sky + game
- `assets/real_pca.json` 3D coords PCA → x,y,z for map
- `assets/real_pca_full.json` tail dims for lab
- `assets/eval_sector_coherence.json`: metrics `knn_sector_purity_at_10` 0.7057 baseline 0.1117 lift 6.32 permutation 0.1106, `cross_ticker` 0.4013 lift 3.59, silhouette -0.0034 vs -0.0204, n=4831/500/11, source `assets/real_data.json`, model `equities_mtnn_v_rebuild_d64_transformer`
- `assets/eval_scoreboard.json` composite + forward gate + hygiene + superseded 0.174 provenance note
- `assets/eval_forward.json` IC and triple-barrier detail (if present)
- `assets/feature_manifest_v6_real.json` 138 feat spec, `tower_list` 17 + wiki
- `assets/manifest.json` PWA manifest — duplicate of top-level `manifest.json` for no-404 parity (copy verified by build check)

Gate tests: `test_eval_sector_coherence.py` PASS purity>0.65, `test_no_ticker_leakage.py` PASS (no ticker, fy 12-d excluded, coverage scalar), `test_no_fabricated_embedding.py` PASS (emb L2≈1, no NaN), `test_verifier_never_fabricates.py` PASS.

## localStorage Keys (aligned with hoops)

- `vh.equities.daily.v2` — {puzzle:number, guesses:[{idx}]} per day, puzzle epoch 2026-08-01
- `vh.equities.streak` — {streak, lastPuzzle}
- `vh.favoriteSector` / `vh.favoriteTicker` — sector tint + confetti primary, fallback legacy `vectorEquities.favoriteTicker`
- `vh.errors` — array max 50 local only no telemetry, quota guard drop oldest half (shared with hoops pattern)
- `vh.vitals` / `vh.vitals-play` — LCP/CLS/INP local only
- `vh.nux-seen`

Legacy: `vectorEquities.v5`, `vectorEquities.pendingLandingGuess` — compat 30 days.

## Cache storage (Cache API via sw.js)

- CACHE_NAME vector-equities-v6-20260809
- CORE immutable cache-first <2MB: manifest, offline.html, shell.css, responsive.css, real_data, pca lite, eval json
- FULL_MTNN lazy: embeddings full f32 if present, archetype_assignments, skill_probe — cached after first fetch SWR
- DENY_CACHE (never-cache): exhaustive SEC xbRL chunks >5MB, .onnx large optional, via fetch network-only 504 on failure
- HTML network-first fallback /offline.html

## Model for fusion

- Blend: (embA+embB)/2 L2-norm → topKForVector nearest real cosine — same as hoops LAB
- Skill blend: avg raw head skills → grade approx round(raw*15+50) 0-99 transparent bar
- Archetype blend: avg logits → softmax 8-way Okabe-Ito bars

## Curation / honesty

- Per-FY z-score prevents era drift; winsor ±4 keeps small-cap spikes but avoids NaN/inf.
- Coverage scalar prevents zero-impute bias seen in early hoops (mask token trick same as hoops MTNN v5 `cat([x·m,m])`).
- Year_norm excluded from X pos-proj gated — mirrors hoops `season_norms.json` μ/σ, no leakage of future year label into tower.
- Placeholder rows: 2026-07-20 S&P 500 expansion filled newly added tickers with sector-centroid+Gaussian noise placeholder embeddings rather than model outputs; diagnostic runs show model-derived and placeholder subsets score similarly on eval, but served matrix 2026-08-05 is 500 tickers v6 real and scores 0.7057/0.4013. Superseded 0.174 (cross 0.167) lift 1.5× from older matrix kept only as provenance note in `eval_scoreboard.json` `superseded`, not shipped metric.
- Engineering metric only — not investment advice, not predictive of returns except IC>0 gate shows embedding geometry knows business future weakly.

Solo personal project, no connection to employer, built with public/free-tier only — 2026-08-09

