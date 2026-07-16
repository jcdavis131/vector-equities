# Vector Equities — Task Plan + Live Todo (FINAL)

Home-only solo project, holistic SEC + NEO + comprehensive pipelines, SOTA multi-task, composite metrics.

## Phase 0 Review & Scaffold [DONE]
- [x] Review hoops train_mtnn 1900 lines families loss weights OneCycle gated fusion
- [x] Scaffold repo pipeline/docs/assets README

## Phase 1 Data Spec & Synthetic [DONE]
17 families 122 feats: income15 balance10 cashflow7 growth9 prof5 lev7 eff5 per_share5 market10 val8 mgmt14 own6 text6 sector3 macro4 form6 bridge2
- [x] build_demo v1 8k recall 0.002
- [x] build_demo v2 continuity 0.88 recall 1.0 sector low
- [x] build_demo v3 continuity 0.80 + sector bias 0.9 archetype 0.8 → 1200x12=14400 sector 86% market 81.6% R2 0.39
- [x] fetch_sec EDGAR CompanyFacts XBRL public domain User-Agent
- [x] fetch_market yfinance free
- [x] parse_neo DEF14A scaffold founder detection

## Phase 2 Skills & Archetypes [DONE]
- [x] 12 Financial Crafts lens
- [x] build_skills 0-100 percentiles per FY
- [x] build_archetypes k-means 8 inertia 147k

## Phase 3 MTNN Model [DONE]
- [x] model.py ResidualTower cat([x·m,m]) 96h→24d skip stacked ResBlocks Gated/Concat/TransformerFusion SkillTowers
- [x] train_mtnn.py Phase B clone rebalanced weights fix epoch0 bug composite proxy 0.5*recall+0.5*purity
- [x] composite_score.py CQS 0.35*recall+0.25*purity+0.20*R2+0.10*sector+0.10*market

## Phase 4 Training & Eval [DONE]
- [x] 14.4k rows 122 feats 17 families 40ep gated d48 b1024 ~90s CPU
- [x] Metrics: sector 86% market 81.6% R2 val0.393 test0.252 purity64% recall 2.8% val tradeoff tunable
- [x] embedding.npz + mtnn_report.json + mtnn_best.pt
- [x] Sweep continuity 0.72-0.88

## Phase 5 Web + GitHub Polish [DONE]
- [x] index.html PCA3 map archetype filter ticker search skill radar similar cos next FY
- [x] docs ARCHITECTURE DETAILED_PLAN HANDOFF PLAN README truthful numbers
- [x] Git repo next: MIT + disclaimer, final review

## Live Log
11:44 start, 11:45 review MTNN, 11:46 scaffold, 11:47 spec+synthetic, 11:48 skills+archetypes, 11:49 train recall 0.002, 11:52 v2 recall 1.0, 11:53 v3 sector 86% market 80% R2 0.40, 11:59 balanced 0.80 run 86%/81.6%/0.39, 12:01 finalize docs dashboard, 12:02 git
