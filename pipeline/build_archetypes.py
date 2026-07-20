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


def build(k=8):
    npz = np.load(DATA_PATH, allow_pickle=False)
    Z = npz["Z"]
    manifest = json.loads(MANIFEST_PATH.read_text())
    # Use game profile features for clustering
    feats = manifest.get("game_features") or manifest["features"][:14]
    idx = [manifest["features"].index(f) for f in feats if f in manifest["features"]]
    X = Z[:, idx]

    km = KMeans(n_clusters=k, n_init=20, random_state=7)
    labels = km.fit_predict(X)

    centroids = km.cluster_centers_

    # save
    out = ROOT / "pipeline" / "data" / "archetype_model.npz"
    np.savez_compressed(
        out, centroids=centroids, labels=labels, names=np.array(ARCHETYPE_NAMES)
    )
    print(f"Archetypes k={k}: built, inertia {km.inertia_:.1f}")
    # overwrite cluster in train_matrix? We'll keep but report
    # update train_matrix.npz cluster field
    npz_data = dict(np.load(DATA_PATH, allow_pickle=False))
    npz_data["cluster"] = labels.astype(np.int64)
    np.savez_compressed(DATA_PATH, **npz_data)
    print(f"Updated cluster in {DATA_PATH}")


if __name__ == "__main__":
    build()
