# DEF14A + Form 4 Chronograph Pipeline — Next Milestone Spec (Top 50 Prototype)

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

