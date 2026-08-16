# Synthetic data audit — vector-equities, 2026-08-15

Triggered by a standing rule: **never synthetic data; full-scale production
implementations for all pipelines and backfills.** Every number below was
measured off `pipeline/data/train_matrix_v5.npz` (4831 × 118, the matrix
`dataset_career.load_bundle()` actually selects) or read out of the builder.

## Scope check first: this is contained to one repo

Same test — columns that are constant where `mask == 1` — across every matrix
in the estate:

| repo | shape | constant-but-observed | never observed |
|---|---|---|---|
| vector-hoops | (12966, 142) | 0 | 0 |
| vector-gridiron | (49860, 85) | 0 | 0 |
| vector-pitch | (2430, 16) | 0 | 0 |
| vector-unified | (4022, 16) | 0 | 0 |
| **vector-equities** | **(4831, 118)** | **23 (19.5%)** | **1** |

## The mechanism

`build_real_v5_career.py:977-983`:

```python
val = row["features"].get(feat_name)
if val is not None:
    Z_raw[i, j] = float(val)
    mask[i, j] = 1.0
```

`mask = 1` is set for any non-`None` value. Every fabricated default is
non-`None`, so **the mask asserts "observed" for values nobody measured**. The
model's mask-aware machinery cannot discount them — this is strictly worse than
honest missingness, because absence is recoverable and a plausible lie is not.

## Site 1 — seeded constants (`:610-630`)

19 variables get a hardcoded seed before any source is consulted. Tracing which
are ever overwritten:

- **Overridden when a DEF 14A record exists (7):** `ceo_comp`, `avg_neo`,
  `ceo_tenure`, `ceo_founder`, `board_size`, `neo_turn`, `insider_net`
- **Never overridden — constant forever (12):** `inst_pct`, `inst_delta`,
  `float_pct`, `top10_conc`, `short_int`, `ceo_age`, `ceo_eq`, `pay_ratio`,
  `board_indep`, `insider_own`, `ceo_pay_vs`, `ceo_dual`

Every company therefore reports `CEO_AGE 55`, `BOARD_INDEP_PCT 75`,
`CEO_PAY_RATIO 200`, `INST_PCT 0.75`, `PIOTROSKI_F_SCORE_PROXY 5`.

## Site 2 — literal substitution, `X if X is not None else <literal>`

Seven sites. Two (`:382` an internal RSI mean, `:1186` a report statistic) are
legitimate. The rest put a plausible number where a measurement was absent:

| line | column | literal | rows affected |
|---|---|---|---|
| 228 | `CEO_TOTAL_COMP` | 12.0 | **4826 / 4831 (99.9%)** |
| 229 | `AVG_NEO_COMP` | 11.0 | **4826 / 4831 (99.9%)** |
| 711, 917 | `PRICE_VS_52W_HIGH` | 0.9 | 511 (10.6%) |
| 712, 918 | `RSI_14_PROXY` | 50 | 517 (10.7%) |
| 1040 | `triple_barrier` | −1 | 511 (10.6%) |

**These are invisible to a constant-column check.** `CEO_TOTAL_COMP` is 99.9%
one literal, but 5 rows differ, so its standard deviation is non-zero and it
does not register as constant. Partial fabrication needs its own test — that is
the main lesson here.

`:228-229` sit **inside `load_def14a()`**, so a filing that parses but whose
compensation table does not yield a number still emits 12.0/11.0. Any backfill
must fix this first or it fabricates over its own new data.

`triple_barrier` is a **label**, not a feature, and it is consumed —
`train_career_mtnn_v6.py:166`, `dataset_career.py:275`. Its −1 sentinel sits
alongside real classes 0 and 1, so 10.6% of the label is invented.

## Site 3 — market static fallback (`:673-698`)

Substitutes `RET_1M/3M/6M = 0.0`, `VOL_30D/90D = 0.0`, `BETA_1Y = 0.0`,
`MOMENTUM_12_1 = 0.0`. Measured: 0–3 rows each, i.e. this path is almost never
taken. Real, but not material today. Worth removing with the rest.

## What is clean

- **`fwd_ret_6m`, the IC metric's target**: 4320 finite rows, 4320 distinct
  values, most-repeated value appears twice. Genuine continuous returns, no
  sentinel flooding. The measured baseline scores against a real target.
- The other four repos, as above.

## Data that already exists and is not wired

| source | size | covers | status |
|---|---|---|---|
| `cache/sec_def14a` | 2.2 GB, 992 filings, 487 tickers | FY2023-24 well, FY2015-22 barely | parser written, **never run to completion** |
| `cache/sec_13f` | 2.8 GB | 2024+ | builder never reads it |
| `cache/sec_form345` | 470 MB, 2015q1+ | **full FY2015-2024 range** | only `fetch_insider_officers.py` reads it |

The builder already prefers `def14a_parsed_v3.jsonl` in a fallback chain; that
file had never been produced, so it silently fell through to a 5-ticker file
from July 30.

**Coverage ceiling for the DEF 14A backfill**, filings mapped to fiscal years:

```
FY2015-2022   1-8 filings each   <2% of rows
FY2023        393 filings        78.6%
FY2024        480 filings        96.0%
reachable     904 / 4831  =  18.7%
```

So the parse fixes governance data for ~19% of rows. **The other 81% only
becomes honest when the mask stops asserting observed.** The mask fix is the
load-bearing change; the parse is the value-add on top.

## Order of work

1. Full DEF 14A parse (running) → real CEO/board data for FY2023-24
2. Remove every literal substitution; unsourced ⇒ `mask = 0`. Existing per-FY
   median fill already handles imputation correctly *with* the mask set right.
3. Wire `sec_form345` (full range) and `sec_13f` (2024+)
4. Rebuild once, re-measure the baseline — this is a new data regime and the
   current 0.5302 anchor will not be comparable

Columns with no available source at all (`float_pct`, `short_int`, `ceo_age`,
`board_indep`, `pay_ratio`, `ceo_dual`, `ceo_eq`, `ceo_pay_vs`) should be
`mask = 0` or dropped from the schema — carrying a dead column is not free, and
that is a design call for the operator.
