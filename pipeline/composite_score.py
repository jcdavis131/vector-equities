"""
Composite Quality Score for Vector Equities — mirrors vector-hoops/pipeline/composite_score.py

Measures:
- recall@10 same-ticker next FY
- cross-cycle archetype purity@20
- sector classification holdout
- next-FY financial profile R2
- market directional accuracy

CQS = 0.35*recall + 0.25*purity + 0.20*next_R2_clipped + 0.10*sector_acc + 0.10*market_bonus
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

def sigmoid(x): return 1/(1+np.exp(-x))

def partial_cqs(recall, purity):
    if recall is None or purity is None:
        return recall or purity or 0.0
    return 0.5*recall + 0.5*purity

def composite_quality(report: dict) -> dict:
    held = report.get("held_out_recall", {}).get("test", {})
    recall = held.get("recall_at_10_mtnn")
    purity = report.get("cross_cycle_archetype_purity_at_20")

    # next profile R2
    nxt = report.get("next_profile", {})
    test_nxt = nxt.get("test", {}) if isinstance(nxt, dict) else {}
    r2 = test_nxt.get("r2") if isinstance(test_nxt, dict) else None

    sector_acc = report.get("sector_top1_acc")
    market_acc = report.get("market_directional_acc")

    # normalize - clip
    r2_clip = float(np.clip(r2, -0.5, 0.9)) if r2 is not None else 0.0
    # market bonus = acc - 0.5 scaled
    market_bonus = float(np.clip((market_acc-0.5)*2, -0.5, 0.5)) if market_acc else 0.0

    parts = []
    if recall is not None: parts.append(recall*0.35)
    if purity is not None: parts.append(purity*0.25)
    parts.append(max(0, r2_clip)*0.20)
    if sector_acc is not None: parts.append(sector_acc*0.10)
    parts.append((0.5+market_bonus)*0.10)  # baseline 0.5

    cqs = float(sum(parts)) if parts else 0.0

    return {
        "cqs": round(cqs,4),
        "recall_at_10": recall,
        "purity_at_20": purity,
        "next_r2": r2,
        "sector_acc": sector_acc,
        "market_acc": market_acc,
        "parts": {"recall": recall, "purity": purity, "r2_clip": r2_clip, "sector": sector_acc, "market_bonus": market_bonus}
    }

def should_promote(report: dict, baseline: float=0.60) -> tuple[bool,str]:
    cq = composite_quality(report)
    cqs = cq["cqs"]
    recall = cq["recall_at_10"] or 0
    # gates from hoops
    if cqs >= baseline + 0.005 and recall >= 0.75:
        return True, f"CQS {cqs} >= {baseline}+0.005 and recall {recall:.3f} OK — promote"
    return False, f"CQS {cqs} below {baseline+0.005} or recall {recall:.3f} weak — needs improvement"

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=str, default="pipeline/data/mtnn_report.json")
    args = ap.parse_args()
    p = Path(args.report)
    if not p.exists():
        print(f"Report {p} not found, run train first")
    else:
        rep = json.loads(p.read_text())
        cq = composite_quality(rep)
        ok, why = should_promote(rep)
        print(json.dumps({"cqs": cq, "promote_ok": ok, "reason": why}, indent=2))
