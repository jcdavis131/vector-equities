#!/usr/bin/env python3
"""A/B the equities MTNN: identical config and seed, ONLY the feature matrix differs.

ARM A  pre-fill matrix   (git HEAD~1)  54.5% observed, 33 zero-coverage columns
ARM B  post-fill matrix                57.0% observed, 29 zero-coverage columns
       the difference is NEO_COUNT, NEO_TURNOVER, CEO_DUALITY, CEO_TENURE

THREE SEEDS PER ARM, because one pair cannot separate a real gain from seed noise. The
spread WITHIN an arm is the noise floor; a between-arm difference smaller than that floor
is not a finding. This is the same discipline the forward probes use — a single split can
be lucky, and so can a single seed.

Config reproduces the shipped model exactly:
    equities_mtnn_v4_transformer_d64_b1  --dim 64 --epochs 60 --fusion transformer
                                         --tower-blocks 1
Shipped reference: test recall@10 0.910, purity@20 0.4855, sector_acc 0.5454,
                   market_acc 0.6581, next_r2 0.1965, CQS 0.6154
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\Users\jcdav\vector-equities")
SC = Path(
    r"C:\Users\jcdav\AppData\Local\Temp\claude\C--Users-jcdav" r"\be69d382-ce38-4d23-b6d1-d92c62546c02\scratchpad\ab"
)
PY = r"C:\Users\jcdav\vector-hoops\pipeline\.venv\Scripts\python.exe"
MATRIX = REPO / "pipeline" / "data" / "train_matrix.npz"
REPORT = REPO / "pipeline" / "data" / "mtnn_report.json"

SEEDS = (7, 11, 13)
ARMS = {"A_prefill": SC / "matrix_OLD.npz", "B_postfill": SC / "matrix_NEW.npz"}

results = []
for arm, src in ARMS.items():
    for seed in SEEDS:
        shutil.copy2(src, MATRIX)
        cmd = [
            PY,
            "pipeline/train_mtnn.py",
            "--epochs",
            "60",
            "--dim",
            "64",
            "--fusion",
            "transformer",
            "--tower-blocks",
            "1",
            "--seed",
            str(seed),
        ]
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print(f"  {arm} seed={seed}  FAILED rc={p.returncode}")
            print((p.stderr or "")[-800:])
            continue
        r = json.loads(REPORT.read_text(encoding="utf-8"))
        row = {
            "arm": arm,
            "seed": seed,
            "test_recall": r["held_out_recall"]["test"]["recall_at_10_mtnn"],
            "val_recall": r["held_out_recall"]["val"]["recall_at_10_mtnn"],
            "purity20": round(r["cross_cycle_archetype_purity_at_20"], 4),
            "sector_acc": round(r["sector_top1_acc"], 4),
            "market_acc": round(r["market_directional_acc"], 4),
            "next_r2_test": r["next_profile"]["test"]["r2"],
            "cqs": r["composite"]["cqs"],
        }
        results.append(row)
        shutil.copy2(REPORT, SC / f"report_{arm}_s{seed}.json")
        print(
            f"  {arm:11} seed={seed:<3} recall={row['test_recall']:.3f} "
            f"purity={row['purity20']:.4f} sector={row['sector_acc']:.4f} "
            f"r2={row['next_r2_test']:.4f} CQS={row['cqs']:.4f}",
            flush=True,
        )

(SC / "ab_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print(f"\nwrote {SC / 'ab_results.json'}  ({len(results)} runs)")
