"""
Build archetypes via k-means on financial profile (like hoops 8 archetypes)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "pipeline" / "data" / "train_matrix.npz"
MANIFEST_PATH = ROOT / "pipeline" / "data" / "feature_manifest.json"

ARCHETYPE_NAMES = [
    "Compounder",
    "Cash_Cow",
    "Turnaround",
    "HyperGrowth_SaaS",
    "Heavy_Industrial",
    "Bank_Capital_Heavy",
    "Moonshot_Bio",
    "Serial_Acquirer",
]


MIN_CLUSTER_COVERAGE = 0.5


def build(k=8):
    npz = np.load(DATA_PATH, allow_pickle=False)
    Z = npz["Z"]
    M = npz["mask"] if "mask" in npz else None
    manifest = json.loads(MANIFEST_PATH.read_text())
    # Use game profile features for clustering, excluding any whose real
    # coverage is too low. Z stores 0.0 for masked-out cells (mask=0), and
    # k-means below reads Z directly with no mask awareness -- a column
    # that's only real for the most recent fiscal years (e.g. a data source
    # with a regulatory effective date) puts a systematic, temporally-
    # concentrated distortion into the clustering rather than uniform noise,
    # which specifically wrecks cross-cycle (cross-year) purity even though
    # the embedding itself may be fine. Measured directly: adding
    # CEO_TOTAL_COMP (19% coverage, concentrated FY2023-24) to this list
    # dropped cross_cycle_archetype_purity_at_20 from 0.48 to 0.39.
    feats = manifest.get("game_features") or manifest["features"][:14]
    dropped = []
    if M is not None:
        kept = []
        for f in feats:
            if f not in manifest["features"]:
                continue
            j = manifest["features"].index(f)
            cov = float(M[:, j].mean())
            if cov < MIN_CLUSTER_COVERAGE:
                dropped.append((f, round(cov, 3)))
                continue
            kept.append(f)
        feats = kept
    if dropped:
        print(f"excluded from clustering (coverage < {MIN_CLUSTER_COVERAGE}): {dropped}")
    idx = [manifest["features"].index(f) for f in feats if f in manifest["features"]]
    X = Z[:, idx]

    km = KMeans(n_clusters=k, n_init=20, random_state=7)
    labels = km.fit_predict(X)

    centroids = km.cluster_centers_

    # save
    out = ROOT / "pipeline" / "data" / "archetype_model.npz"
    np.savez_compressed(out, centroids=centroids, labels=labels, names=np.array(ARCHETYPE_NAMES))
    print(f"Archetypes k={k}: built, inertia {km.inertia_:.1f}")
    # overwrite cluster in train_matrix? We'll keep but report
    # update train_matrix.npz cluster field
    npz_data = dict(np.load(DATA_PATH, allow_pickle=False))
    npz_data["cluster"] = labels.astype(np.int64)
    np.savez_compressed(DATA_PATH, **npz_data)
    print(f"Updated cluster in {DATA_PATH}")


if __name__ == "__main__":
    build()
