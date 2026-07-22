# Vector Equities — Company Embedding Explorer

Live: [equities.dumbmodel.com](https://equities.dumbmodel.com) · GitHub: [jcdavis131/vector-equities](https://github.com/jcdavis131/vector-equities)

> Solo personal project, no connection to employer, built with public/free-tier only.
> **Built in raw WebGPU / WebGL / Canvas — no Unity/Unreal, just browser graphics APIs straight.** Zero engine, raw `<canvas>` + custom shaders, static Vercel.

2741 company-FYs · 283 tickers · 17 family towers · 64-d transformer MTNN · 8 archetypes · 11 GICS sectors

Interactive PCA map of public companies. Each company-year is embedded via 17 residual towers (income, balance, cashflow, growth, profitability, leverage, efficiency, per-share, market, valuation, management, ownership, disclosure, sector, macro, form, bridge) fused by a 4-layer transformer into a 64-d L2-normalized vector. Cosine similarity = business similarity.

- **Data:** SEC EDGAR XBRL CompanyFacts (2015-2024) + market data + V2 10-K chunks (3439 tickers, ~150k chunks, Item 1/1A/7 with tables)
- **Model:** 17× ResidualTower → Transformer fusion (d_model 128, 4L 4H) → 64-d + wiki embedding tower 384-d MiniLM
- **Training:** Same-ticker adjacent FY contrastive (InfoNCE) + sector hard negatives
- **Frontend:** `index.html` loads `assets/real_data.json` (2741 points, 12 skills, PCA xyz) + `chunks_v2/{TICKER}.json` wiki with tables



## Quickstart

```bash
python3 pipeline/fetch_sec_summary.py --limit 300
python3 pipeline/build_real_from_summary.py --limit 300
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py
python3 pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128
python3 pipeline/regen_assets.py
```

## Architecture

- **Towers:** 17 families, `cat([x·m, m]) → 96h → 24d` with skip
- **Fusion:** attention over towers, FY embedding, CLS token → 64-d L2
- **Heads:** 8 archetypes, 11 sectors, 14-d profile, next-year profile, 12 skill grades, valuation, market
- **Skills:** Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Momentum, Management Quality, Yield, Disclosure

## Assets

- `assets/manifest.json` — rows, tickers, dim, towers
- `assets/real_data.json` — 2741 points with xyz, skills, emb
- `assets/real_data_latest.json` — latest FY per ticker (283)
- `assets/real_pca.json` — PCA projection
- `assets/eval_sector_coherence.json` — sector-coherence eval of the published embedding (k-NN sector purity@10, cosine silhouette) vs the random-assignment expectation given sector sizes; regenerate with `python pipeline/eval_sector_coherence.py`, gated by `tests/test_eval_sector_coherence.py`. Engineering metric of the embedding geometry only — not investment advice, not predictive of returns.

## Deploy

Vercel static import, domains: `equities.dumbmodel.com` and `equities.jcamd.com` (redirect via `vercel.json`).

© 2026 Vector Equities · [equities.dumbmodel.com](https://equities.dumbmodel.com)
