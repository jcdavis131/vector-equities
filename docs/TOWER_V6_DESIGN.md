# Tower V6 Design — Holistic Market View

## Goal
Add 3 new towers before unfreezing backbone and retraining with w_f6=5 + ranking to push fwd std toward optimal 4%.

Current v5: 17 families, D=122, N=2741, IC fwd 0.16 after rank tuning.

Proposed v6: 20 families, D~154-160, same N, 3 new towers.

## Tower 1: industry_event (Industry News & Events)
Purpose: Capture sector-specific shocks, disruption, regulatory shifts.

**Data sources (free / public):**
- GDELT 2.0 Doc API + GKG (no key): query per sector/year for themes: STRIKE, RECALL, LAWSUIT, REGULATORY, MERGER, PRODUCT_LAUNCH, SUPPLY_CHAIN
- SEC 8-K filing counts per sector/year (material events, Item 1.01, 2.01, 7.01) — already fetched in exec registry
- EDGAR full text sentiment for industry keywords (proxy for MDA sentiment but sector-aggregated)

**Features (10):**
1. IND_NEWS_VOL_Z: log count GDELT docs sector/year, z-scored vs sector history
2. IND_NEWS_TONE_AVG: GDELT avg tone (-10 to +10) -> normalized
3. IND_NEG_EVENT_CNT: strike+recall+lawsuit+layoff themes count
4. IND_POS_EVENT_CNT: product launch, contract, patent
5. IND_REGULATORY_RISK: regulation+antitrust themes
6. IND_MA_INTENSITY: M&A theme volume YoY
7. IND_SUPPLY_DISRUPTION: supply chain + shortage + logistics themes
8. IND_EARN_BREADTH: sector % beating estimates (from existing EARN_SURPRISE_STREAK aggregated)
9. IND_DISPERSION_MOM: std of RET_12M within sector
10. IND_VOL_SPIKE: VIX-like sector vol / market vol

Implementation: `pipeline/towers_v6/industry_gdelt.py` queries GDELT per sector (GICS -> keyword map), aggregates yearly, caches JSON.

Fallback if offline: use sector_context + form features as proxy with added noise.

## Tower 2: political_risk (Political News & Events, Elections Worldwide)
Purpose: Elections, policy uncertainty, geopolitics.

**Sources:**
- GPR Index (Iacoviello): https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls — monthly, global
- EPU (Baker-Bloom-Davis): https://www.policyuncertainty.com — US, Global, China daily -> FY avg
- NY Fed GSCPI is in trade tower but also political
- Election calendar: built static CSV from Wikipedia + IFES, feature: months to next election per country
- World Bank WGI Political Stability, but annual — add as slow-moving
- ACLED (optional) for conflict count, but heavy

**Features (10):**
1. GPR_GLOBAL_AVG_FY: Geopolitical Risk avg
2. GPR_YOY: change
3. EPU_US_AVG_FY
4. EPU_GLOBAL_AVG_FY
5. ELEC_PROX_US: 12 / (months_to_US_presidential +1) -> 1 near election
6. ELEC_PROX_GLOBAL: revenue-weighted exposure to upcoming elections (use sector exposure: e.g., Energy high EM election exposure)
7. TARIFF_RISK: Trade Policy Uncertainty subindex (from EPU)
8. WGI_POL_STABILITY: World Bank score for US (slow)
9. GOV_SHUTDOWN_PROX: US debt ceiling / shutdown search volume proxy (GDELT GOV theme)
10. RATE_VOL_3M: std of 10Y yield last 90 days

Many are market-wide (same for all tickers in a FY), but provide regime conditioning — this lets model learn 2024 bearish mean 3.3% was driven by high GPR/EPU, not collapse signal. Solves earlier quantile mapping failure.

Implementation: `towers_v6/political.py` downloads GPR, EPU CSVs, caches.

## Tower 3: global_trade_commodity (Global Trade & Raw Materials)
Purpose: Freight, dollar, commodities drive margins, especially Energy, Materials, Industrials.

**Sources (all free, no key needed via yfinance as proxy for FRED):**
- yfinance: CL=F (WTI), BZ=F (Brent), HG=F (Copper), HRC proxy: BHP, Ali? Actually use STEEL ETF SLX, Lumber: LBS=F, NatGas: NG=F, Corn: C=F, Wheat: W=F, DXY: DX-Y.NYB, Baltic Dry: use BDRY ETF or ^DJI proxy? Use ^BDI via FRED fallback, Freightos FBX: use BDRY
- NY Fed GSCPI: https://www.newyorkfed.org/medialibrary/research/gscpi/gscpi_data.xlsx
- FRED alternatives via yfinance: DXY, BALTIC via symbol BDIY? Use BDRY (Breakwave Dry Bulk)
- USD/CNY: CNY=X

**Features (12):**
1. OIL_WTI_YOY: (CL=F FY avg YoY)
2. OIL_BRENT_SPREAD: Brent-WTI
3. COPPER_YOY: HG=F
4. STEEL_PROXY_YOY: SLX ETF
5. LUMBER_YOY: LBS=F
6. NATGAS_YOY: NG=F
7. DXY_YOY: dollar strength
8. USDCNY_YOY: trade tension
9. BDRY_YOY: freight proxy
10. GSCPI_AVG_FY: supply chain pressure (standardized)
11. COMMODITY_BETA_X_SECTOR: interaction: commodity YoY * sector sensitivity (from sector_context)
12. AGRI_YOY: equal weight corn+wheat

All YoY computed vs prior FY, then z-scored 3-year rolling.

**Normalization:** Z-score per feature vs 5-year rolling to avoid lookahead, then fill NaN with 0, mask accordingly.

## Model Integration

Current `model_career.py`:
- `fam_dims` dict auto-detects families, so adding 3 new entries automatically creates new ResidualTower per family.
- Fusion: ContinuousFusion attends over n_towers (20 vs 17) — will learn to gate political/trade towers higher in high VIX years.

No code change needed except feature_manifest update.

Training plan with new towers:
- D_new ~ 154, same N 2741
- w_f6=5, w_dd=1.5, w_vol=0.3, w_entry=2.0, w_nce=1.0, rank_w=1.0, var_w=0.2
- Loss: MSE fwd + ranking loss (256 pairs, margin 0.02, threshold true diff>5%) + var loss (std_true - std_pred)^2 *0.2
- Optimizer Adam 2e-3, 30 epochs, early stop on VAL IC 6M
- Goal: push std from 0.7% toward optimal 4% = IC*true_std. With IC 0.16, target std ~4%, currently 0.7% is 5.7x collapsed. Ranking loss should expand.


Memory: New matrix 2741x154 ~ 1.7MB, fine for 7.8G.

