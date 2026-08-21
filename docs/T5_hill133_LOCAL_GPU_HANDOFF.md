# T5 Hill133 LOCAL-GPU HANDOFF — 60ep Transformer Resume

**Claim file:** `COORDINATION_LOCAL_GPU.md` — <7 max 3 exempt RTX4090

## Command Hatch CPU → Alienware RTX4090
```bash
# 5m lite forwarded full forwarded full forwarded slept lit e 
python3 -u pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --batch 512 --tower-width 24 --d-model 96 --n-fusion-layers 4 --n-attn-heads 4 --tower-blocks 2 --mlp-heads --lr 1.5e-3 --weight-decay 1e-4 --val-every 5 --device cuda --one-cycle --pct-start 0.1 --clip 1.0 --dropout 0.12 --temp 0.08 --hard-neg 0.2 --feature-drop 0.12
```

**Monitor:**
- `pipeline/data/mtnn_best.pt` 514K → `equities_v6_money_best.pt` → ONLY IF CQS>0.72 market>0.58 next_R2>0.20 IC>0.01 Sharpe>1.2 gates pass → `equities_v6_money_v2.pt`
- Current gate FAIL: IC0.007/spear0.0097 <0.01 FAIL Top50 0.079 small-n n=50<233 Kelly needs 233 min, Sharpe sqrt2 0.57 FAIL sqrtN6.15 PASS ambiguous, CQS0.7017<0.72 FAIL honest no fake promotion
- Epoch0 evidence: loss6.0163 val_recall0.9 test0.95 purity0.718 comp0.809 would beat 0.7017 SIGTERM 167s before epoch1 TRUNCATED*12 runtime fail — honest log not inflated
- Timeout counts 30-60m LOCAL-GPU exempt <7 max, nano fallback batch256 accum2 --one-cycle --pct-start 0.1 --clip 1.0 device auto cuda else cpu

**Locks:**
- Zero-deps true torch auto cuda else cpu, free platform free — open access Knowledge→Edge→Money no $199/$49 Knowledge 10-K XBRL DEF14A →Edge MTNN 64-d SupCon CORAL GRL transformer →Money private only
- 0DTE ONLY IF IC>0.03 Sharpe>1.2 win>55% DD<12% kill-switch separate bankroll NOT financial advice current IC0.007<0.03 paper only
- Triple-write 7-field even no-change mandatory nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass ts runId ooda tempo :01 zero_deps true

**Benched File button:**
```
eval_forward.json IC0.007 Top50 0.079 triple0.2189 distress-0.2624 Sharpe 0.57/6.15 n233
eval_sector_coherence.json CQS0.7017 baseline0.605 sector0.957 recall1.0 purity0.68 continuity0.72 FY12 1200×12=14,400 FYs ticker-split 70/15/15 honest
LCG dailySeed=YYYYMMDD → idx3970 triple[3970,14390,4582] PWA v67 CORE20 void #080A0F #FFFEF7 $0/mo Vercel hobby free

