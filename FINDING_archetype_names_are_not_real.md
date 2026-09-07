# The eight equities "archetypes" are k-means cluster ids wearing invented names

Found 2026-09-07 while replacing the map's generated points with the real export.

## What the code does

`pipeline/build_archetypes.py` clusters the real financial feature matrix:

    km = KMeans(n_clusters=k, n_init=20, random_state=7)
    labels = km.fit_predict(X)
    np.savez_compressed(out, centroids=..., labels=labels, names=np.array(ARCHETYPE_NAMES))

`ARCHETYPE_NAMES` is a hardcoded list at the top of the same file. Cluster index `i` gets
`ARCHETYPE_NAMES[i]`. No step anywhere inspects a cluster's membership and names it from
what is inside. `README.md` states this plainly - "8 k-means" - but the exported field is
called `archetype` and the names read as financial descriptions, so every downstream
consumer treats them as meaning something.

## What the data says

Cross-tab of archetype against sector, all 4,831 rows:

| archetype (cluster) | n | top sector | share in top sector |
|---|---|---|---|
| HyperGrowth_SaaS | 1984 | Financials | 23% |
| Bank_Capital_Heavy | 935 | Industrials | 27% |
| Cash_Cow | 449 | Technology | 16% |
| Moonshot_Bio | 445 | Technology | 22% |
| Serial_Acquirer | 328 | Consumer Discretionary | 28% |
| Heavy_Industrial | 325 | Technology | 31% |
| Compounder | 228 | Technology | 30% |
| Turnaround | 137 | Technology | 20% |

Every name is contradicted by its own membership:

* **Bank_Capital_Heavy** holds 256 Industrials rows and 45 Financials.
* **HyperGrowth_SaaS** is 41% of the entire dataset and its largest sector is Financials,
  with 244 Utilities and 219 Real Estate rows.
* **Heavy_Industrial** is mostly Technology; only 33 of its 325 rows are Industrials.
* **Moonshot_Bio** has 44 Healthcare rows out of 445.

The most typical FY2024 member of each cluster, by distance to that cluster's mean 64-d
embedding, makes it concrete:

    HyperGrowth_SaaS   -> Dominion Energy (Utilities)
    Bank_Capital_Heavy -> Dover Corporation (Industrials)
    Heavy_Industrial   -> Monolithic Power Systems (Technology)
    Moonshot_Bio       -> Vertiv (Industrials)
    Compounder         -> Huntington Bancshares (Financials)

No cluster exceeds 31% concentration in its top sector, so these are not sector proxies
that got mislabelled. They are unnamed clusters.

## Why it matters

The clusters are real: k-means on a real feature matrix is a real computation, and the
purity metric built on them measures something. The **names** are not real. Publishing a
map that colours a regulated utility "HyperGrowth_SaaS" ships a fabrication wrapped in
genuine data, which is the failure mode that is hardest to catch precisely because every
number around it is sound.

## What this repo should do

1. Stop exporting invented names as `archetype`. Export `cluster: 0..7` and let a consumer
   that wants names derive them from membership.
2. If names are wanted, generate them from each cluster's own centroid in feature space -
   the top few features by absolute z - rather than from a hardcoded list.
3. The public page must not display these names. It colours by **sector**, which is a real,
   externally verifiable attribute of every row.

Nothing was changed in the pipeline for this finding. It is a report.
