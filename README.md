# Vector Equities — Company Embedding Explorer

Live: [equities.dumbmodel.com](https://equities.dumbmodel.com) · GitHub: [jcdavis131/vector-equities](https://github.com/jcdavis131/vector-equities)

> Solo personal project, no connection to employer, built with public/free-tier only.

An interactive PCA map of public companies. Each company fiscal-year is embedded by a multi-tower neural net — 17 residual towers over statement families (income, balance, cashflow, growth, profitability, leverage, efficiency, per-share, market, valuation, management, ownership, disclosure, sector, macro, form, bridge) fused by a 4-layer transformer into a 64-dim L2-normalized vector — and cosine distance in that space is the site's notion of business similarity. Static site (plain HTML/JS/canvas, no framework), hosted on Vercel.

The served dataset currently covers ~2,700 company-FYs across ~280 tickers (2015–2024); exact counts live in `assets/real_data.json` and `assets/manifest.json` and change as the pipeline reruns.

- **Data:** SEC EDGAR XBRL CompanyFacts (2015–2024) + market data + 10-K text chunks (Item 1/1A/7, tables included)
- **Model:** 17× ResidualTower → transformer fusion (d_model 128, 4 layers, 4 heads) → 64-d, plus a 384-d MiniLM wiki-text tower
- **Training:** same-ticker adjacent-FY contrastive (InfoNCE) with sector hard negatives
- **Frontend:** `index.html` loads `assets/real_data.json` (points with xyz, 12 skill grades, embeddings) and per-ticker 10-K chunk files

## Quickstart

```bash
python3 pipeline/fetch_sec_summary.py --limit 300
python3 pipeline/build_real_from_summary.py --limit 300
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py
python3 pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128
python3 pipeline/regen_assets.py
```

## Architecture

- **Towers:** 17 families, `cat([x·m, m]) → 96h → 24d` with skip connections
- **Fusion:** attention over towers + FY embedding + CLS token → 64-d L2-normalized
- **Heads:** 8 archetypes, 11 GICS sectors, 14-d profile, next-year profile, 12 skill grades, valuation, market
- **Skills:** Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Momentum, Management Quality, Yield, Disclosure

## Evaluation

`assets/eval_sector_coherence.json` measures label coherence of the published embedding geometry: k-NN sector purity@10 of 0.174 (cross-ticker 0.167) against a random-assignment baseline of 0.112 — a 1.5–1.6x lift, measured on the matrix as served (the file records provenance caveats, including placeholder rows from the S&P 500 expansion). Regenerate with `python pipeline/eval_sector_coherence.py`; gated by `tests/test_eval_sector_coherence.py`. This is an engineering metric of the embedding geometry only — not investment advice, and not predictive of returns.

## Deploy

Vercel static import; domains `equities.dumbmodel.com` and `equities.jcamd.com` (redirect via `vercel.json`).

MIT. Solo personal project, no connection to employer, built with public/free-tier only.
