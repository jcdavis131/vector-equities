# Wrapper that forces v6 manifest
from pathlib import Path

from pipeline.dataset_career import *

manifest_path = Path("pipeline/data/feature_manifest_v6.json")
if manifest_path.exists():
    print(f"Using v6 manifest {manifest_path}")
