"""Run the vector-bench multi-target gauntlet on the real equities dataset.

For each registry target (forward_return, forward_realized_vol,
drawdown_exceedance) this script:

1. builds a leakage-safe temporal BenchmarkTask from bench/data/equities_bench_v1.npz
   (features standardized with TRAIN-ONLY statistics),
2. trains the repo's real MTNN — pipeline/model.py EquitiesMTNN family towers +
   gated fusion trunk — ONCE, jointly on all three forward targets, on fiscal
   years 2015-2020 with fy2021 as the early-stopping validation year (both
   inside the harness train side; test years 2022-2024 are never touched),
3. slots the trained model's test-row predictions in as the MTNN rung next to
   the full default prediction ladder, and
4. writes the schema-1.1 domain report to bench/benchmark_report.json.

Seeded and CPU-only. Reproduce:
    python bench/fetch_prices.py --out-dir /tmp/equities_prices
    python bench/build_dataset.py --prices-dir /tmp/equities_prices
    python bench/run_benchmark.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vector_bench.baselines import MTNNRung  # noqa: E402
from vector_bench.registry import get_domain_spec  # noqa: E402
from vector_bench.report import write_domain_report  # noqa: E402
from vector_bench.runner import run_domain_benchmark  # noqa: E402
from vector_bench.tasks import build_task_for_target  # noqa: E402

from pipeline.model import EquitiesMTNN  # noqa: E402 - the repo's real architecture

SEED = 42
TEST_CUT = 2022
VAL_YEAR = 2021
D_TOWER = 24
D_EMB = 48
BATCH = 512
MAX_EPOCHS = 300
PATIENCE = 30
WEIGHT_DECAY = 1e-4

# Small documented hyperparameter grid; the winner is chosen on the fy2021
# validation score (mean of per-target val metrics), never on test rows.
GRID = [
    {"lr": 1.5e-3, "ret_weight": 1.0},
    {"lr": 1.5e-3, "ret_weight": 2.0},
    {"lr": 5e-4, "ret_weight": 1.0},
    {"lr": 5e-4, "ret_weight": 2.0},
]

TARGETS = ("forward_return", "forward_realized_vol", "drawdown_exceedance")


def standardize_train_only(X_raw, X_mask, harness_train_rows):
    """Impute + standardize with statistics from harness-train rows only."""
    X = X_raw.astype(np.float64).copy()
    X[X_mask <= 0] = np.nan
    med = np.nanmedian(X[harness_train_rows], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for j in range(X.shape[1]):
        col = X[:, j]
        col[np.isnan(col)] = med[j]
    mu = X[harness_train_rows].mean(axis=0)
    sd = X[harness_train_rows].std(axis=0)
    sd[sd < 1e-9] = 1.0
    X = np.clip((X - mu) / sd, -5.0, 5.0)
    return X.astype(np.float32)


class ForwardHeadsMTNN(nn.Module):
    """The repo's EquitiesMTNN trunk (family towers + gated fusion) with one
    fresh linear head per wired forward target — the multi-task model under test."""

    def __init__(self, fam_dims: dict[str, int]):
        super().__init__()
        self.trunk = EquitiesMTNN(
            fam_dims,
            n_seasons=1,  # season embedding disabled: test years are unseen at fit time
            d_tower=D_TOWER,
            d_emb=D_EMB,
            n_skills=0,
            fusion_mode="gated",
        )
        self.head_ret = nn.Linear(D_EMB, 1)
        self.head_vol = nn.Linear(D_EMB, 1)
        self.head_dd = nn.Linear(D_EMB, 1)

    def forward(self, xs, ms, season_ids):
        emb = self.trunk.encode(xs, ms, season_ids)
        return (
            self.head_ret(emb).squeeze(-1),
            self.head_vol(emb).squeeze(-1),
            self.head_dd(emb).squeeze(-1),
        )


def family_tensors(X, M, families):
    fams = sorted(set(families))
    cols = {f: np.where(families == f)[0] for f in fams}
    xs = {f: torch.tensor(X[:, cols[f]]) for f in fams}
    ms = {f: torch.tensor(M[:, cols[f]]) for f in fams}
    fam_dims = {f: len(cols[f]) for f in fams}
    return xs, ms, fam_dims


def masked_mse(pred, y, m):
    if m.sum() == 0:
        return torch.zeros((), dtype=pred.dtype)
    return ((pred[m] - y[m]) ** 2).mean()


def masked_bce(logit, y, m):
    if m.sum() == 0:
        return torch.zeros((), dtype=logit.dtype)
    return nn.functional.binary_cross_entropy_with_logits(logit[m], y[m])


def _val_score(pr, pv, pd_, y, m, rows):
    """Validation model-selection score: mean of per-target val metrics
    (spearman IC for the two regressions, ROC-AUC for the drawdown flag).
    Computed on fy2021 only — test rows are never touched here."""
    from vector_bench.metrics import roc_auc as _auc
    from vector_bench.metrics import spearman_ic as _ic

    r = rows
    parts = []
    mr = m["ret"][r].numpy()
    parts.append(_ic(y["ret"][r].numpy()[mr], pr.numpy()[mr]))
    mv = m["vol"][r].numpy()
    parts.append(_ic(y["vol"][r].numpy()[mv], pv.numpy()[mv]))
    md = m["dd"][r].numpy()
    parts.append(_auc(y["dd"][r].numpy()[md], pd_.numpy()[md]))
    return float(np.mean(parts))


def train_one(d, X, dd_threshold, lr, ret_weight, log):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    fy = d["time_key"]
    families = d["feature_families"].astype(str)
    xs, ms, fam_dims = family_tensors(X, d["X_mask"].astype(np.float32), families)
    season = torch.zeros(len(fy), dtype=torch.long)

    # target tensors (scaled on MTNN-train rows only)
    grad_rows = np.where(fy <= VAL_YEAR - 1)[0]
    val_rows = np.where(fy == VAL_YEAR)[0]

    y = {}
    m = {}
    scale = {}
    for name, key in (("ret", "y_forward_return"), ("vol", "y_forward_realized_vol")):
        yv = d[key].astype(np.float64)
        mask = ~np.isnan(yv)
        tr_vals = yv[grad_rows][mask[grad_rows]]
        mu, sd = float(tr_vals.mean()), float(tr_vals.std() + 1e-9)
        scale[name] = (mu, sd)
        yv_s = (np.where(mask, yv, 0.0) - mu) / sd
        y[name] = torch.tensor(yv_s, dtype=torch.float32)
        m[name] = torch.tensor(mask)
    dd = d["forward_max_drawdown"].astype(np.float64)
    dd_mask = ~np.isnan(dd)
    y["dd"] = torch.tensor(np.where(dd_mask, dd <= dd_threshold, 0.0), dtype=torch.float32)
    m["dd"] = torch.tensor(dd_mask)

    model = ForwardHeadsMTNN(fam_dims)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    def batch_loss(rows):
        pr, pv, pd_ = model({f: xs[f][rows] for f in xs}, {f: ms[f][rows] for f in ms}, season[rows])
        return (
            ret_weight * masked_mse(pr, y["ret"][rows], m["ret"][rows])
            + masked_mse(pv, y["vol"][rows], m["vol"][rows])
            + masked_bce(pd_, y["dd"][rows], m["dd"][rows])
        )

    def forward_rows(rows):
        model.eval()
        with torch.no_grad():
            r = torch.tensor(rows)
            return model({f: xs[f][r] for f in xs}, {f: ms[f][r] for f in ms}, season[r])

    rng = np.random.default_rng(SEED)
    best_score, best_state, best_epoch, since = -np.inf, None, -1, 0
    val_rows_t = torch.tensor(val_rows)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(grad_rows)
        for s in range(0, len(perm), BATCH):
            rows = torch.tensor(perm[s : s + BATCH])
            opt.zero_grad()
            loss = batch_loss(rows)
            loss.backward()
            opt.step()
        pr, pv, pd_ = forward_rows(val_rows)
        score = _val_score(pr, pv, pd_, y, m, val_rows_t)
        if score > best_score + 1e-5:
            best_score, best_epoch, since = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    log(f"  lr={lr} ret_weight={ret_weight}: best val score {best_score:.4f} " f"at epoch {best_epoch}")

    def predict(rows):
        pr, pv, pd_ = forward_rows(rows)
        return (
            pr.numpy() * scale["ret"][1] + scale["ret"][0],
            pv.numpy() * scale["vol"][1] + scale["vol"][0],
            torch.sigmoid(pd_).numpy(),
        )

    return predict, best_epoch, best_score


def train_mtnn(d, X, dd_threshold, log):
    """Train the grid, select the config on the fy2021 validation score."""
    best = None
    for cfg in GRID:
        predict, ep, score = train_one(d, X, dd_threshold, cfg["lr"], cfg["ret_weight"], log)
        if best is None or score > best[2]:
            best = (predict, ep, score, cfg)
    predict, ep, score, cfg = best
    log(f"selected config {cfg} (val score {score:.4f}, best epoch {ep})")
    return predict, ep, score, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "bench" / "data" / "equities_bench_v1.npz")
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "benchmark_report.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    fy = d["time_key"]
    harness_train = fy <= VAL_YEAR
    X = standardize_train_only(d["X"], d["X_mask"], harness_train)

    # drawdown threshold: harness-train median (never sees test labels)
    dd = d["forward_max_drawdown"].astype(np.float64)
    dd_mask = ~np.isnan(dd)
    dd_threshold = float(np.median(dd[harness_train & dd_mask]))
    print(f"drawdown_exceedance threshold (train median forward max drawdown): {dd_threshold:.4f}")

    predict, best_epoch, best_val, best_cfg = train_mtnn(d, X, dd_threshold, log=print)

    spec = get_domain_spec("equities")
    y_by_target = {
        "forward_return": d["y_forward_return"].astype(np.float64),
        "forward_realized_vol": d["y_forward_realized_vol"].astype(np.float64),
        "drawdown_exceedance": np.where(dd_mask, (dd <= dd_threshold).astype(float), np.nan),
    }
    pred_col = {"forward_return": 0, "forward_realized_vol": 1, "drawdown_exceedance": 2}

    training_note = (
        f"MTNN = pipeline/model.py EquitiesMTNN (family towers d_tower={D_TOWER}, "
        f"gated fusion, d_emb={D_EMB}, n_seasons=1) + 3 linear forward heads, "
        f"trained jointly (ret_weight*mse + mse + bce) on fy2015-2020; model "
        f"selection (early stop, patience {PATIENCE}, + grid {GRID}) on the fy2021 "
        f"validation score (mean of val spearman_ic/spearman_ic/roc_auc); selected "
        f"{best_cfg} at epoch {best_epoch} (val score {best_val:.4f}); AdamW "
        f"wd={WEIGHT_DECAY} batch={BATCH} seed={SEED}. Test rows (fy>=2022) never "
        "seen in training or model selection."
    )

    tasks = {}
    mtnns = {}
    for t in TARGETS:
        yv = y_by_target[t]
        rows = np.where(~np.isnan(yv))[0]
        task = build_task_for_target(
            spec.target(t),
            domain="equities",
            X=X[rows],
            y=yv[rows],
            group_key=d["ticker"][rows],
            time_key=fy[rows],
            time_cut=TEST_CUT,
            seed=SEED,
            extra_notes={
                "dataset": str(args.data.name),
                "n_rows": str(len(rows)),
                "horizon_actual": (
                    "126 trading days (~6 months) from the July-1 fy+1 anchor — "
                    "the repo's fwd6 convention; the registry's '1m' horizon label "
                    "is prose, the split enforces leakage safety"
                ),
                "dd_threshold": f"{dd_threshold:.4f}" if t == "drawdown_exceedance" else "",
                "training": training_note,
            },
        )
        split = task.make_split()
        test_rows_global = rows[split.test_idx]
        preds = predict(test_rows_global)[pred_col[t]]
        tasks[t] = task
        mtnns[t] = MTNNRung(predictions=preds)
        print(f"{t}: n={len(rows)} train={len(split.train_idx)} test={len(split.test_idx)}")

    dsc = run_domain_benchmark(spec, tasks, mtnns)
    path = write_domain_report(dsc, args.out)
    print(f"\n{dsc.aggregate['headline']}")
    for ts in dsc.targets:
        if ts.scorecard is None:
            print(f"  {ts.target_name}: {ts.status} ({ts.note})")
            continue
        v = ts.scorecard.verdicts.get(ts.primary_metric)
        print(
            f"  {ts.target_name} [{ts.primary_metric}]: mtnn={v.mtnn_value:.4f} "
            f"best_baseline={v.best_baseline}={v.best_baseline_value:.4f} "
            f"delta={v.mtnn_delta:+.4f} beats={v.mtnn_beats_best_baseline}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
