#!/usr/bin/env bash
# run_local_gpu.sh — Equities v6 MTNN — local GPU easy pickup
# Hatch VM = CPU (no CUDA), Alienware = GPU when available
set -euo pipefail
EPOCHS="${1:-60}"
cd "$(dirname "$0")/.."
echo "[equities] epochs=$EPOCHS $(date -u)"

DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then DEVICE="cuda"; echo "[equities] CUDA -> $DEVICE"; else echo "[equities] -> cpu"; fi

if [ ! -f pipeline/data/train_matrix.npz ] && [ ! -f pipeline/data/train_matrix_real.npz ]; then
  echo "[equities] Missing pipeline/data/train_matrix.npz (needs SEC 4831 FYs) - Alienware prep only"
  echo "[equities] Try: PYTHONPATH=../../vector-hub/packages/vector-core/src python3 pipeline/build_real_v6_towers.py"
  exit 0
fi

# equities train_mtnn.py has no --device flag, auto device inside
PYTHONPATH=../../vector-hub/packages/vector-core/src:./src:$PYTHONPATH python3 pipeline/train_mtnn.py --epochs "$EPOCHS" --dim 64 2>&1 | tee -a pipeline/cache/train_equities_${EPOCHS}ep.log || echo "[equities] graceful exit - see log"

echo "[equities] done"
