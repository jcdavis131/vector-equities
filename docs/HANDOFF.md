# Vector Equities — HANDOFF 2026-08-09

Current state:

- 4,831 company-FYs (500 tickers 2015-2024), 64-d MTNN 17 towers transformer fusion, sector purity 0.7057 lift 6.32x, cross-ticker 0.4013 lift 3.59x, eval_sector_coherence.json provenance honest 2026-08-05.

- Model: 17× ResidualTower 96h→24d skip, d_model 128 4L 4H CLS→64 L2, 384-d MiniLM wiki tower optional.

- Gated by tests/test_eval_sector_coherence.py (>0.65 purity) + test_no_ticker_leakage.py (FY 12-d excluded, coverage scalar mean mask, year_norm excluded).

- Pipeline: fetch_sec_summary → build_real_from_summary → build_skills + archetypes → train_mtnn → regen_assets → eval_sector_coherence + eval_forward.

- Running: python3 pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128; assets/real_data.json committed static host; client-side inference optional.

- Open: DEF 14A Neo parser full (DEF14A_CHRONOGRAPH_SPEC), insider form 4 sequence transformer, MD&A Loughran-McDonald sentiment free HF MiniLM.

- Verify: python -m json.tool assets/eval_scoreboard.json && python -m json.tool assets/eval_sector_coherence.json && python -m pytest tests/test_eval_sector_coherence.py tests/test_no_ticker_leakage.py -q

- License MIT.

