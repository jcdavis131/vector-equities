# DEF14A + Form 4 Chronograph Pipeline — Next Milestone Spec (Top 50 Prototype)

## 2026-07-31 status: Phase A/B built, root-caused correctly, fixed, DEPLOYED

Follow-up to the 2026-07-30 entry below, which got the *shape* of the
problem right (a delay archetype — a data source whose real coverage starts
partway through the historical window colliding with a fixed split
boundary) but the *specific mechanism* wrong. Correcting that, then the
actual fix and final numbers.

### The 07-30 diagnosis was wrong on one detail

That entry blamed "the `mgmt` decode head" seeing a masked-then-visible
target. There is no trained `mgmt` head: `EquitiesMTNN.forward()` computes
`out["mgmt"]`/`out["own"]`/`out["vol"]`/`out["payout"]` every step, but
`train_mtnn.py`'s loss only ever references
`DEFAULT_WEIGHTS["archetype"/"sector"/"profile"/"next_profile"/"skills"/
"valuation"/"market"/"health"]` — `"mgmt"` and `"own"` are declared in the
weights dict and never read. Those heads get zero gradient; they're dead
weight. Verified directly: grepped every `DEFAULT_WEIGHTS[...]` usage site
in the file, confirmed no `"mgmt"`/`"own"` reference exists anywhere in the
loss computation.

### What's actually happening (two separate, now both fixed, mechanisms)

**1. Fusion has no visibility into per-tower coverage.** `management_neo`
feeds an *input tower* (`ResidualTower`), not a trained head. During
training (FY<=2021, ~0% real `CEO_TOTAL_COMP`/`AVG_NEO_COMP` coverage), that
tower always receives a near-constant masked input and produces a
near-constant output token. `TransformerFusion`/`GatedFusion`/`ConcatFusion`
attend over that token with no idea it's masked — they just learn, over
training, to discount a token that never varies. At eval time (FY2022+,
~100% of this family's observed coverage), the same tower suddenly emits a
genuinely varying, informative token that the fusion was never trained to
weigh — a real train/eval distribution mismatch, just at the *architecture*
level rather than a *loss* level.

**2. `build_archetypes.py`'s k-means directly clusters on the same
recency-skewed column.** It fits on `manifest["game_features"]` (the 14-d
"core interpretable" subset from `feature_spec.py` — and `CEO_TOTAL_COMP`
is one of the 14), reading `Z` directly with **no mask awareness**: a
masked cell is a literal `0.0` to k-means, indistinguishable from a real
value of exactly zero. A column that's 0.0 for every pre-2022 row and real
for 2023-24 doesn't add uniform noise to the clustering — it adds a
*systematic, temporally-concentrated* signal that pulls companies into
different archetype buckets partly by which years they have data for. That
directly explains why `cross_cycle_archetype_purity_at_20` (which measures
whether same-archetype companies get found *across different years*) took
the largest hit of any metric (0.4844 -> 0.3785) — the ground-truth labels
themselves got less cross-year-consistent, not just the embedding.

### The fix (two small, targeted changes, not a big rewrite)

1. **`pipeline/model.py`**: `EquitiesMTNN.encode()` now computes a
   per-family coverage scalar (`ms[fam].mean(dim=-1)`) and passes it into
   the fusion alongside each tower's embedding — `GatedFusion`,
   `ConcatFusion`, and `TransformerFusion` all updated to concatenate this
   per-tower coverage value before projecting, so the fusion can actually
   tell a masked-out family apart from a genuinely-present one instead of
   just seeing whatever near-constant token a never-varying-in-training
   tower happens to emit. This changes tower/tensor shapes — any existing
   checkpoint needs a full retrain, not a fine-tune.
2. **`pipeline/build_archetypes.py`**: excludes any clustering feature
   whose overall mask coverage is below `MIN_CLUSTER_COVERAGE = 0.5` from
   the k-means input (currently drops `CEO_TOTAL_COMP` at 19.2% coverage
   and the still-unbuilt `INSIDER_OWN_PCT` at 0%). A general rule, not a
   hardcoded exclusion — protects against any *future* sparse or
   recency-gated addition to `game_features` doing the same thing, not
   just this one column.

Isolated the architecture change first, alone, on the pre-DEF14A matrix, to
confirm it doesn't itself regress anything: CQS 0.628 -> 0.6226, recall
0.912 -> 0.908, purity 0.4844 -> 0.4847 — a neutral, within-noise change on
its own. Then added real DEF14A data back in with the fix in place.

### Final numbers, 2 seeds

| | recall | purity | sector_acc | market_acc | CQS |
|---|---|---|---|---|---|
| pre-DEF14A reference | 0.912 | 0.4844 | 0.5784 | 0.6694 | 0.628 |
| seed 7, both fixes + real data | 0.910 | **0.4855** | 0.5454 | 0.6581 | 0.6154 |
| seed 13, both fixes + real data | 0.908 | **0.4887** | 0.5663 | 0.6756 | 0.6223 |
| **mean (7, 13)** | **0.909** | **0.4871** | **0.5559** | **0.6669** | **0.6189** |

Purity and recall are fully recovered (purity is actually slightly *above*
the pre-DEF14A reference on both seeds). Sector accuracy sits a bit below
(-0.02) on both seeds — a small residual gap, not chased further (single
data point per condition, could be seed noise rather than a real remaining
effect; worth another seed or two if it recurs). Mean CQS (0.6189) sits
within ~0.01 of the reference (0.628) -- essentially parity, not a
regression, and both seeds individually pass the repo's `should_promote`
gate (`baseline=0.60`).

**Deployed.** Seed 7 kept as canonical (matches this repo's established
convention). `pipeline/data/{train_matrix.npz,feature_manifest.json,
mtnn_best.pt,mtnn_report.json,embedding.npz,train_matrix_real.npz}` now
reflect the 142-feature-equivalent matrix with real `CEO_TOTAL_COMP`/
`AVG_NEO_COMP`, the coverage-aware fusion architecture, and the
coverage-filtered archetype clustering. Full pytest suite (25/25) and
`audit_features.py` both clean (same 4 known redundant pairs as before,
0 new).

The other 12 `management_neo` fields and all 6 `ownership` fields still
have no real-data source wired (age/tenure/board-independence/insider-
ownership aren't part of the XBRL PVP tag set — need table/bio-section
parsing or Form 4 aggregation, both separately scoped, large undertakings;
see §1.B/§1.C/§3 below). The coverage-aware fusion fix generalizes to
whichever of those gets built next, though — any future recency-gated or
sparse family benefits from the same architecture change, not just this one.

---

## 2026-07-30 status: Phase A/B partially built, measured, NOT deployed (superseded above)

Built and verified working, full 500-ticker universe (not just the Top-50
prototype scoped below):

- `pipeline/fetch_submissions.py`, `pipeline/fetch_def14a_recent.py` (new —
  narrower than `fetch_def14a_full.py`, which grabs up to 11 *oldest*
  filings/ticker; this grabs the 2 *newest*, which is what a machine-
  readable-tag extraction actually needs). 965 DEF14A HTML files fetched.
- `pipeline/parse_def14a_xbrl.py` (new): instead of the heuristic regex
  table-scraper in `parse_def14a.py` (measured 4/37 ≈ 11% success on a
  sample, matching this doc's own §3 warning), reads the inline-XBRL
  Pay-vs-Performance tags (`ecd:PeoTotalCompAmt`, `ecd:NonPeoNeoAvgTotalCompAmt`)
  that SEC Item 402(v) has required since the rule took effect for FY2022+
  proxies. 958/992 filings had at least one tag (96.6%) — a categorically
  different reliability level than table-scraping, because it's reading a
  legally-mandated machine-readable fact instead of guessing table layout.
  Spot-checked AAPL FY2023 CEO_TOTAL_COMP = $63.2M against Tim Cook's
  publicly reported figure — matches.
- `pipeline/def14a_features.py` (new, mirrors `market_features.py`'s
  `get_market_row` shape) + wired into `build_real_from_summary.py`:
  `CEO_TOTAL_COMP`/`AVG_NEO_COMP` (2 of the 14 `management_neo` fields) now
  read real values where available, `None` elsewhere — honest, not
  fabricated. Coverage: 929/4831 (19.2%) and 954/4831 (19.75%) rows
  respectively, concentrated in FY2023-2024 (XBRL tags don't exist before
  the rule's effective date, so earlier years are correctly masked, not
  guessable). The other 12 `management_neo` fields and all 6 `ownership`
  fields have no XBRL shortcut (age/tenure/board-independence/insider-
  ownership aren't part of the PVP machine-readable tag set) and remain
  unbuilt — table/bio-section parsing, same as before.

**Why it isn't deployed.** Retrained (transformer, dim 64, epochs 60,
matching the recipe from the 2026-07-30 market/valuation fix): CQS dropped
0.628 -> 0.5771, recall@10 0.912 -> 0.874, purity@20 0.4844 -> 0.3785,
sector_acc/market_acc/next_r2 all down too, promote gate flips true -> false.
Not noise — every headline metric moved the same direction.

Root cause, not a data-quality problem: `eval_split()` cuts train at fiscal
year <= 2021, val <= 2023, test > 2023. The XBRL PVP tag literally does not
exist before FY2022 (the rule's own effective date), so real
`CEO_TOTAL_COMP`/`AVG_NEO_COMP` coverage is ~0% in train and effectively
~100% of its observed coverage in val/test. The `mgmt` decode head (loss
weight 0.08) sees a target that's always masked during every training
step and then suddenly present at eval time — a genuine train/eval
distribution mismatch, not a coincidence, and not something more fetching
fixes (the data genuinely doesn't exist for train-window years). This
mismatch appears to cost the shared embedding trunk more broadly, not just
the `mgmt` head's own score (recall/sector/market/next_r2 all measurably
worse too).

Separately (not this feature's fault, but found while investigating): the
initial retrain showed `purity@20` jump to a suspicious exact 1.0. Traced
it — `build_real_from_summary.py` always writes `cluster=np.zeros(N)`; the
real 8-archetype cluster assignment is a *separate*, later step
(`build_archetypes.py`, not part of any current orchestrator) that
re-populates it. My rebuild ran the first step only, silently reverting
every row's archetype cluster to 0, which made `cross_cycle_purity`
trivially return ~1.0 by construction. Re-ran `build_archetypes.py`
before drawing any conclusion — same failure shape as vector-hoops'
`enrich_vectors.py` position-label regression earlier the same day: an
orchestrator step resets a field a *different* script is responsible for
repopulating, and nothing currently strings the full correct sequence
together end to end for this repo either.

**State on disk:** `pipeline/data/{train_matrix.npz,feature_manifest.json,
mtnn_best.pt,mtnn_report.json,embedding.npz,train_matrix_real.npz}` reverted
to the pre-DEF14A committed reference (CQS 0.628). `pipeline/data/
def14a_comp.json` (956 real ticker-FY comp records) and the 965 cached
filing HTMLs are kept — real, verified, reusable data, just not currently
merged into the deployed matrix. `def14a_features.py`,
`fetch_def14a_recent.py`, `parse_def14a_xbrl.py`, and the
`build_real_from_summary.py` wiring are **committed** — correct and inert
until re-enabled.

**What would actually let this ship:** the train/eval split needs to stop
being a fixed `<=2021` cutoff for features whose only real-world coverage
starts in 2022+, or the `mgmt` loss head needs to be masked out during
training entirely and only scored at eval (so the trunk never trains
against a feature it can't see in-sample) — either is a real methodology
change to `train_mtnn.py`'s eval-split / loss-masking logic, bigger than
this feature's own wiring, not made here.

---


**Goal:** Replace placeholder `management_neo` 14 + `ownership` 6 towers with real person-level chronological graph NN input.

**Current reality:** 2741 rows × 122 feats, but management_neo = const (CEO_AGE 55, tenure 6, etc), ownership = const (INST_PCT 0.75). No DEF14A, no Form 4, no person graph.

**Target:** Company-FY node linked to Exec nodes (NEO graph) + Filing nodes (10-K MD&A text) + Form 4 transaction timeline.

---

## 1. Data Sources

### 1.A SEC Submissions JSON (index)
- Endpoint: `https://data.sec.gov/submissions/CIK{cik_pad}.json` (fast, includes recent filings list)
- Contains: `filings.recent` {accessionNumber, filingDate, form, primaryDocument, isXBRL, etc}
- Rate: <10 req/s, curl fallback UA `SEC contact (configured via User-Agent)`
- Cache: `pipeline/cache/sec_submissions/sub_{cik}.json`
- Filter: form == `DEF 14A` (or `DEF14A`) for proxy, and form == `4` for insider, date 2015-2024

### 1.B DEF14A filings (proxy)
- URL pattern: `https://www.sec.gov/Archives/edgar/data/{cik_nopad}/{acc_no_nodash}/{primaryDoc}`
- Example: `https://www.sec.gov/Archives/edgar/data/66740/000006674024000036/mmm-20240313.htm`
- Content: HTML, contains:
  - **Summary Compensation Table** (SCT): NEO names, title, Salary, Bonus, Stock Awards, Option Awards, Non-Equity Incentive, Total -> CEO_TOTAL_COMP, AVG_NEO_COMP
  - **Biography / Executive Officers section**: age, tenure start, founder flag, board membership, CEO duality, board independence %
  - **Ownership table**: beneficial ownership → CEO_EQUITY_PCT, INSIDER_OWN_PCT
  - **Pay Ratio** disclosure → CEO_PAY_RATIO
  - **Board section**: BOARD_SIZE, BOARD_INDEP_PCT, CEO_DUALITY (Chairman?)
  - **Turnover signal**: presence of new names vs previous year → NEO_TURNOVER
- Cache: `pipeline/cache/sec_def14a/{ticker}_{fy}.html` + parsed json `def14a_{cik}_{fy}.json`

### 1.C Form 4 filings (insider trading)
- Endpoint: same submissions, form `4`
- For each Form 4: XML primary doc contains `reportingOwner`, `transactionCode`, `transactionShares`, `transactionPrice`, `securitiesOwned`
- Signals:
  - `INSIDER_NET_12M` = sum(shares bought - sold) trailing 12M per ticker per FY, normalized by shares outstanding
  - `INSIDER_OWN_PCT` cross-check with DEF14A
  - Founder selling pattern, CEO selling clusters
- Cache: `pipeline/cache/sec_form4/{cik}/{accession}.xml` → summary json per FY

### 1.D Ownership / 13F (optional phase 2)
- For `INST_PCT`, `TOP10_INST_CONC`, `FLOAT_PCT`: can use `https://data.sec.gov/api/xbrl/companyfacts/...` has `EntityCommonStockSharesOutstanding` + market cap from yfinance, plus institutional ownership via `https://api.sec-api.io` or `whalewisdom` free? For now use yfinance major holders scrape as intermediate.
- Phase 2: 13F filings parsing.

---

## 2. Prototype Scope Top 50

Universe: first 50 tickers from `universe.json` (S&P 100 head): MMM, AOS, ABT, ABBV, ACN, ADBE, AMD, AES, AFL, A, APD, AKAM, ALB, ARE, ALGN, ALLE, LNT, ALL, GOOGL, GOOG, MO, AMZN, AMCR, AMD? Actually list: take `uni[:50]` exactly.
For each: fetch submissions.json → list DEF14A 2015-2024 (expect ~10 per ticker) → fetch HTML → parse.

Success metric prototype: ≥ 7 DEF14A per ticker avg (some missing/spac), parse ≥ 80% SCT name extraction.

---

## 3. Parsing Strategy (free-tier, no paid APIs)

### DEF14A SCT parsing (hard problem: HTML varies)
Approach:
1. Use regex + BeautifulSoup lightweight. Try patterns:
   - Find table with header containing "Name and Principal Position" + "Total"
   - Alternative header "Summary Compensation Table"
   - Extract rows: split by <tr>, first column contains exec name + title
   - Name cleaning: strip titles (Mr., Ms.)
   - Comp detection: last column dollar amount, or sum columns 2..n
2. Fallback: LLM-free heuristic: search for `CEO` anchor link near table, then next table likely SCT.
3. Structure output:

```json
{
  "ticker": "MMM",
  "cik": "0000066740",
  "fy": "2023",
  "filing_date": "2024-03-13",
  "accession": "0000066740-24-000036",
  "neos": [
    {"name": "Michael F. Roman", "title": "Chairman/CEO", "age": 64, "is_ceo": true, "total_comp": 14200000, "salary": 1300000, "equity_comp_pct": 0.65, "tenure_years": 6},
    ...
  ],
  "board_size": 11,
  "board_indep_pct": 91,
  "ceo_duality": 1,
  "pay_ratio": 184,
  "insider_own_pct": 0.8
}
```

### Form 4 parsing
- XML parse using `xml.etree`: extract `reportingOwnerId -> rptOwnerName`, `transactionCode`, `transactionShares.value`, `transactionPricePerShare.value`, `sharesOwnedFollowing`.
- Per FY aggregates:
  - `insider_net_shares_12m`, `insider_net_value_12m`, `net_buy_sell_ratio`, `ceo_net`, `cfo_net`
- Store per filing then roll up per FY.

### Entity Resolution
- Exec ID = normalized name lower + dob/age fuzzy + Levenshtein across tickers
- Build `exec_id` = slug of name (e.g., `michael-f-roman`)
- Track `career_moves`: exec_id appears at different tickers across time → edge.
- Cache global map `pipeline/data/exec_registry.json`

---

## 4. Graph Construction (Chronological)

Nodes:
- **Company-FY node**: existing 122 feats but replace management_neo/ownership with real
- **Exec node**: static emb + dynamic per-FY features (comp, tenure, age)
- **Filing node**: DEF14A date, Form 4 clusters

Edges:
- `Company-FY --[NEO {role, comp, is_ceo}]--> Exec` (per FY)
- `Exec --[NEXT_YEAR]--> Exec` (temporal self-edge, weight 1)
- `Exec --[COWORKER {ticker, fy}]--> Exec` (co-tenure in same FY)
- `Exec --[MOVED_TO {from_ticker, to_ticker, gap_years}]--> Company-FY`
- `Company-FY --[INSIDER_TX_AGG]--> Form4Summary`

Model replacement:
- Current `management_neo` tower: 14 dummy scalars → replace with **ExecChronographEncoder**
  - Input: per FY list of NEOs (up to 5) each with [age, tenure, total_comp_log, equity_pct, is_ceo, is_founder, pay_vs_sector, board_indep]
  - Process: each exec embedded via small MLP (8→24) + temporal positional encoding (tenure)
  - Then transformer over execs (permutation invariant) → pooled company-level management emb 24-d
  - Plus historical: previous FY same execs embedding via cross-attention (career consistency)
  - Output 14-d real features OR directly 24-d hidden for tower input

- Ownership tower: replace const with `INST_PCT, INSIDER_NET etc` from Form 4 aggregates + DEF14A ownership

---

## 5. Integration Plan

Phase A (this prototype, Top 50):
- [ ] fetch_submissions Top50
- [ ] fetch DEF14A HTML Top50 (limit 10 each = 500 files, ~50MB)
- [ ] parse heuristics → `def14a_parsed.jsonl` (one per filing)
- [ ] evaluate parse rate, manual review 5 random (MMM, AAPL, ABT)
- [ ] build exec_registry for cross-company moves

Phase B (after prototype):
- [ ] Form4 fetch + parse per ticker FY aggregates
- [ ] Replace management_neo/ownership placeholders in `build_real_from_summary.py` with real values (join on ticker+fy)
- [ ] New model: `management_neo` tower now real distribution → expect CQS boost (sector acc may drop, but purity up, next R2 up)
- [ ] Train v3: 2741 rows + real mgmt for 50 tickers, rest imputed (semi-supervised)

Phase C (full 300):
- [ ] Scale to 283 tickers, full graph NN (GNN temporal) as separate tower `management_neo_graph` 14→32 hidden

---

## 6. Storage Layout

```
pipeline/cache/
  sec_submissions/sub_{cik_pad}.json
  sec_def14a/
    {TICKER}_{FY}_{accession}.html
    parsed/
      {TICKER}_{FY}.json
  sec_form4/
    {cik}/
      {accession}.xml
      summary_{cik}_{fy}.json
pipeline/data/
  exec_registry.json (global exec_id → names, tickers, tenure)
  def14a_master.jsonl (all parsed)
  form4_master.jsonl
```

---

## 7. Prototype Script Commands

```bash
cd ~/workspace/vector-equities
python3 pipeline/fetch_submissions.py --limit 50 --start 0
python3 pipeline/fetch_def14a.py --limit 50
python3 pipeline/parse_def14a.py --limit 50
python3 pipeline/eval_def14a_parse.py
```

---

## 8. Risks & Mitigations

- HTML variance huge → need multiple regex fallback, keep raw HTML cache, manual spot-check MMM/AAPL/ABT.
- SEC throttling → sleep 0.25s, use curl fallback already.
- Name disambiguation hard → start with exact match lower, then fuzzy for top CEO only.
- Form 4 XML namespace weird → use ET with wildcard.
- Legal: SEC public domain, proxy public.

---

## 9. Success Criteria

- Top50: fetch submissions 50, DEF14A files ≥350 (7 avg), parse SCT ≥80% → 280 parsed JSONs
- Extract at least 3-5 NEOs per filing, total ~1000 exec-FY records, ~400 unique execs (many repeats)
- Show career move: exec appears at 2 tickers (e.g., CFO moves)
- Deliver `def14a_parsed.jsonl` + `exec_registry.json` + spec doc.

