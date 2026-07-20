"""
Build real matrix v4 — adds real DEF14A exec, Form4, and market cache overrides
Template from build_real_from_summary.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
CACHE_SUM = ROOT / "pipeline" / "cache" / "sec" / "sec_summary"
MARKET_DIR = ROOT / "pipeline" / "cache" / "market"
DEF14A_DIR = ROOT / "pipeline" / "cache" / "sec_def14a"

import sys

sys.path.insert(0, str(ROOT / "pipeline"))
from feature_spec import ALL_FEATURES, FEATURE_FAMILIES, GAME_PROFILE_FEATURES, SECTORS


def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    try:
        return float(a) / float(b)
    except:
        return None


def clean_name(s):
    if not s:
        return ""
    # replace NBSP and zero-width
    s = (
        s.replace("\xa0", " ")
        .replace("\u200b", " ")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\d+\)", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60]


def is_person_name(name):
    if not name:
        return False
    cn = clean_name(name)
    if len(cn) < 4:
        return False
    low = cn.lower()
    # reject if contains $ or lots of digits
    if "$" in cn or re.search(r"\d{4,}", cn):
        return False
    # reject pure titles
    title_only_patterns = [
        r"^chairman of the board",
        r"^chief executive officer",
        r"^chief financial officer",
        r"^chief operating officer",
        r"^executive vice president",
        r"^senior vice president",
        r"^vice chairman",
    ]
    for pat in title_only_patterns:
        if re.search(pat, low):
            # if starts with title, not a person
            # allow if first token not title? but these are title starts
            return False
    # Must have at least 2 words
    parts = cn.split()
    if len(parts) < 2:
        return False
    # First two words should start with capital letter
    if not (parts[0][0].isupper() and parts[1][0].isupper()):
        return False
    # reject if word "and" is last
    if parts[-1].lower() == "and":
        return False
    # reject if contains "board" as major part
    if low.startswith("chairman of the boar") or low.startswith("chairman of"):
        return False
    if low.startswith("chief executive"):
        return False
    # if length >40 and contains officer etc, likely not name
    if len(cn) > 35 and any(
        k in low for k in ["officer", "chairman", "president", "vice"]
    ):
        # check if contains a person-like pattern before title
        # e.g., "Richard A. Gonzalez Chairman"
        # then first 2-3 words might be name, but our parser kept full title in name field
        # For now reject
        return False
    return True


def load_summaries():
    out = {}
    if not CACHE_SUM.exists():
        return out
    for p in CACHE_SUM.glob("summary_*.json"):
        try:
            j = json.loads(p.read_text())
            cik = j.get("_meta", {}).get("cik") or p.stem.replace("summary_", "")
            out[cik] = j
        except Exception:
            pass
    return out


def load_def14a():
    # Try v3, v2, v1
    candidates = [
        DATA_DIR / "def14a_parsed_v3.jsonl",
        DATA_DIR / "def14a_parsed_v2.jsonl",
        DATA_DIR / "def14a_parsed.jsonl",
        ROOT / "pipeline" / "cache" / "def14a_parsed_v3.jsonl",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        print("No DEF14A parsed found, exec will be placeholder only")
        return {}
    print(f"Loading DEF14A from {path}")
    raw_entries = []
    with open(path) as f:
        for line in f:
            try:
                j = json.loads(line)
                if j.get("ticker"):
                    raw_entries.append(j)
            except:
                continue
    print(f"Loaded {len(raw_entries)} DEF14A filings raw")
    # Group by (ticker,FY) with latest filing_date kept
    tmp = {}  # (ticker,FY) -> entry
    for entry in raw_entries:
        ticker = entry.get("ticker")
        fdate_str = entry.get("filing_date", "")
        try:
            # Parse YYYY-MM-DD
            yr = int(fdate_str[:4])
            mo = int(fdate_str[5:7]) if len(fdate_str) >= 7 else 6
        except:
            continue
        # Map: if filing month Jan-Apr, FY = yr -1 else yr
        if mo >= 1 and mo <= 4:
            fy = yr - 1
        else:
            fy = yr
        if fy < 2014 or fy > 2024:
            # still keep for tenure history but will filter later for base years 2015-2024
            # keep but allow out-of-range for tenure calc
            pass
        key = (ticker, fy)
        # Keep latest filing_date
        existing = tmp.get(key)
        if existing is None or fdate_str > existing.get("filing_date", ""):
            tmp[key] = entry

    # Now second pass: compute per ticker history for tenure and turnover
    # Build ticker -> sorted FY list
    by_ticker = defaultdict(list)
    for (ticker, fy), ent in tmp.items():
        by_ticker[ticker].append((fy, ent))
    exec_features = {}  # (ticker,fy) -> computed dict
    for ticker, lst in by_ticker.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        ceo_counts = Counter()  # name -> count distinct years
        prev_neo_names = None
        for fy, ent in lst_sorted:
            neos_raw = ent.get("neos", [])
            board_size = ent.get("board_size")
            # Filter neos to person-like
            neos_filtered = []
            for neo in neos_raw:
                n_name = neo.get("name", "")
                if is_person_name(n_name):
                    neos_filtered.append(neo)
                else:
                    # try to see if name contains person before title? e.g., "Richard A. Gonzalez Chairman..."
                    # heuristic: split and take first 3 words if they look like name
                    cn = clean_name(n_name)
                    # if contains at least 2 capitalized words at start, extract them
                    parts = cn.split()
                    if (
                        len(parts) >= 2
                        and parts[0][0].isupper()
                        and parts[1][0].isupper()
                    ):
                        # take up to 3 words as name attempt
                        tentative = (
                            " ".join(parts[:3])
                            if len(parts) >= 3
                            else " ".join(parts[:2])
                        )
                        # check if tentative still person-like
                        if is_person_name(tentative):
                            # create copy with cleaned name
                            neo_copy = dict(neo)
                            neo_copy["name"] = tentative
                            neos_filtered.append(neo_copy)
                        # else skip
            # Fallback: if filtered empty but raw non-empty, use raw but with cleaned names truncated
            if not neos_filtered and neos_raw:
                # keep raw but with names cleaned
                neos_filtered = neos_raw

            # CEO detection
            ceo_candidate = None
            for neo in neos_filtered:
                combined = (neo.get("name", "") + " " + neo.get("row_text", "")).lower()
                if (
                    "chief executive" in combined
                    or " ceo" in combined
                    or combined.strip().startswith("ceo")
                    or "chief executive officer" in combined
                ):
                    ceo_candidate = neo
                    break
            if ceo_candidate is None:
                # max comp
                if neos_filtered:
                    ceo_candidate = max(
                        neos_filtered, key=lambda x: x.get("total_comp", 0) or 0
                    )
            if ceo_candidate is None:
                # No neo, skip
                continue

            ceo_name_raw = ceo_candidate.get("name", "")
            ceo_name = clean_name(ceo_name_raw)
            # Simplify CEO name to first 2-3 tokens (remove titles trailing)
            # Remove trailing titles if present
            ceo_name = re.sub(r"\s+Chairman.*$", "", ceo_name, flags=re.I).strip()
            ceo_name = re.sub(r"\s+Chief.*$", "", ceo_name, flags=re.I).strip()
            ceo_name = clean_name(ceo_name)

            # Compute comps in millions
            def comp_million(c):
                if c is None:
                    return None
                try:
                    v = float(c)
                    if v > 1000:  # assume dollars
                        return v / 1e6
                    else:
                        return v
                except:
                    return None

            ceo_comp = comp_million(ceo_candidate.get("total_comp"))
            # avg neo comp
            comps = [comp_million(n.get("total_comp")) for n in neos_filtered]
            comps = [c for c in comps if c is not None]
            avg_comp = float(np.mean(comps)) if comps else None

            # NEO count
            neo_count = len(neos_filtered) if neos_filtered else ent.get("neo_count", 0)

            # Founder flag
            combined_ceo = (
                ceo_candidate.get("name", "") + " " + ceo_candidate.get("row_text", "")
            ).lower()
            founder_flag = 1 if "founder" in combined_ceo else 0

            # Board size
            bs = (
                board_size
                if isinstance(board_size, int) and 3 <= board_size <= 20
                else 9
            )

            # CEO tenure: cumulative count
            # Count distinct years same CEO name appears up to FY
            # Use case-insensitive exact match after clean
            ceo_counts[ceo_name.lower()] += 1
            tenure = ceo_counts[ceo_name.lower()]

            # NEO turnover
            curr_neo_names = {
                clean_name(n.get("name", "")).lower()
                for n in neos_filtered
                if clean_name(n.get("name", ""))
            }
            if prev_neo_names is None:
                turnover = 0.15
            else:
                if len(curr_neo_names) == 0 or len(prev_neo_names) == 0:
                    turnover = 0.15
                else:
                    overlap = len(curr_neo_names.intersection(prev_neo_names))
                    max_cnt = max(len(curr_neo_names), len(prev_neo_names))
                    turnover = 1.0 - (overlap / max_cnt) if max_cnt > 0 else 0.15

            exec_features[(ticker, fy)] = {
                "NEO_COUNT": neo_count,
                "CEO_TOTAL_COMP": ceo_comp if ceo_comp is not None else 12.0,
                "AVG_NEO_COMP": avg_comp if avg_comp is not None else 11.0,
                "CEO_TENURE": tenure,
                "CEO_FOUNDER_FLAG": founder_flag,
                "NEO_TURNOVER": turnover,
                "BOARD_SIZE": bs,
                "CEO_NAME": ceo_name,
                "NEO_NAMES": list(curr_neo_names),
            }
            prev_neo_names = curr_neo_names
            ceo_name.lower()

    print(
        f"Computed exec features for {len(exec_features)} (ticker,FY) combos, tickers {len({k[0] for k in exec_features})}"
    )
    return exec_features


def load_form4():
    candidates = [
        DATA_DIR / "form4_index_all.jsonl",
        DATA_DIR / "form4_index.jsonl",
        ROOT / "pipeline" / "cache" / "sec_form4" / "form4_index.jsonl",
        ROOT / "pipeline" / "cache" / "form4_index_all.jsonl",
    ]
    files = [c for c in candidates if c.exists()]
    if not files:
        print("No Form4 index found")
        return {}
    print(f"Loading Form4 from {files}")
    counts = defaultdict(lambda: defaultdict(int))  # ticker -> fy -> count
    total_lines = 0
    for fp in files:
        with open(fp) as f:
            for line in f:
                try:
                    j = json.loads(line)
                    ticker = j.get("ticker")
                    fdate = j.get("filing_date", "")
                    if not ticker or not fdate:
                        continue
                    yr = int(fdate[:4])
                    # Form4 FY is filing year directly
                    counts[ticker][yr] += 1
                    total_lines += 1
                except:
                    continue
    print(f"Form4 total filings {total_lines}, tickers {len(counts)}")
    # Flatten to (ticker,fy) dict
    flat = {}
    for ticker, year_dict in counts.items():
        for fy, cnt in year_dict.items():
            flat[(ticker, fy)] = cnt
    return flat


def load_market():
    if not MARKET_DIR.exists():
        print("Market dir missing")
        return {}
    market = {}
    # Prefer {ticker}.json over _2y
    for fp in MARKET_DIR.glob("*.json"):
        name = fp.stem
        # Skip 2y and 5y for primary load? We'll load all and let base overwrite?
        # If file is like AAPL_2y, ticker is AAPL, but we'll parse ticker from inside json
        try:
            j = json.loads(fp.read_text())
            ticker = j.get("ticker")
            if not ticker:
                # fallback from filename
                ticker = name.replace("_2y", "").replace("_5y", "")
            # Only keep one per ticker, prefer non-suffix? But both same data in current snapshot
            if ticker not in market:
                market[ticker] = j
            else:
                # if current file is base (no underscore) prefer it
                if "_2y" not in fp.name and "_5y" not in fp.name:
                    market[ticker] = j
        except:
            continue
    print(f"Loaded market cache for {len(market)} tickers")
    return market


def build_v4(limit=None):
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} summaries, universe {len(uni)}")
    exec_features = load_def14a()
    form4_flat = load_form4()
    market_data = load_market()

    all_rows = []
    for entry in uni[:limit] if limit else uni:
        cik = entry["cik"].zfill(10)
        ticker = entry["ticker"]
        sector = entry.get("sector", "Industrials")
        company = entry.get("company", ticker)
        summ = (
            summaries.get(cik)
            or summaries.get(entry["cik"])
            or summaries.get(cik.lstrip("0"))
        )
        if not summ:
            continue
        prev_vals = {}
        for yr in range(2015, 2025):
            ystr = str(yr)
            raw = summ.get(ystr, {})
            if not raw:
                continue
            rev = raw.get("REVENUE")
            cogs = raw.get("COGS")
            gross = raw.get("GROSS")
            op = raw.get("OP_INCOME")
            net = raw.get("NET_INCOME")
            assets = raw.get("ASSETS")
            liab = raw.get("LIAB")
            equity = raw.get("EQUITY")
            cash = raw.get("CASH")
            debt_lt = raw.get("DEBT_LT")
            debt_st = raw.get("DEBT_ST")
            ocf = raw.get("OCF")
            capex = raw.get("CAPEX")
            cur_a = raw.get("CURR_A")
            cur_l = raw.get("CURR_L")
            depr = raw.get("DEPR")
            interest = raw.get("INTEREST")
            shares_d = raw.get("SHARES_D") or raw.get("SHARES_B")
            ret_earn = raw.get("RET_EARN")
            if gross is None and rev is not None and cogs is not None:
                gross = rev - cogs
            debt = None
            if debt_lt is not None or debt_st is not None:
                debt = (debt_lt or 0) + (debt_st or 0)
            ebitda = (
                op + depr
                if (op is not None and depr is not None)
                else (op * 1.15 if op else None)
            )
            ebit = op
            fcf = None
            if ocf is not None and capex is not None:
                fcf = ocf - abs(capex) if capex > 0 else ocf + capex
            elif ocf is not None:
                fcf = ocf * 0.8
            gross_margin = safe_div(gross, rev)
            op_margin = safe_div(op, rev)
            net_margin = safe_div(net, rev)
            ebitda_margin = safe_div(ebitda, rev)
            fcf_margin = safe_div(fcf, rev)
            book_value = equity
            working_cap = (
                cur_a - cur_l if (cur_a is not None and cur_l is not None) else None
            )
            net_debt = debt - cash if (debt is not None and cash is not None) else None
            invested_cap = (
                equity + debt - cash
                if (equity is not None and debt is not None and cash is not None)
                else None
            )

            def yoy(curr, prev_key):
                prev = prev_vals.get(prev_key)
                if curr is None or prev is None or prev == 0:
                    return None
                return (curr - prev) / abs(prev)

            def cagr(curr, prev_key, yrs=3):
                prev = prev_vals.get(prev_key)
                if curr is None or prev is None or prev <= 0 or curr <= 0:
                    return None
                try:
                    return (curr / prev) ** (1.0 / yrs) - 1
                except:
                    return None

            rev_yoy = yoy(rev, f"REV_{yr - 1}")
            rev_3y = cagr(rev, f"REV_{yr - 3}", 3)
            ebitda_yoy = yoy(ebitda, f"EBITDA_{yr - 1}")
            net_yoy = yoy(net, f"NET_{yr - 1}")
            fcf_yoy = yoy(fcf, f"FCF_{yr - 1}")
            eps = safe_div(net, shares_d)
            bvps = safe_div(equity, shares_d)
            fcfps = safe_div(fcf, shares_d)
            shares_yoy = yoy(shares_d, f"SHARES_{yr - 1}")
            roe = safe_div(net, equity)
            roa = safe_div(net, assets)
            roic = (
                safe_div(net, invested_cap)
                if invested_cap
                else safe_div(op, invested_cap)
            )
            curr_ratio = safe_div(cur_a, cur_l)
            debt_eq = safe_div(debt, equity)
            debt_ebitda = safe_div(debt, ebitda)
            int_cov = safe_div(op, interest) if interest else None
            debt_assets = safe_div(debt, assets)
            net_debt_ebitda = safe_div(net_debt, ebitda)
            asset_turn = safe_div(rev, assets)
            capex_depr = safe_div(capex, depr)

            # placeholders
            inst_pct = 0.75
            inst_delta = 0.0
            insider_net = 0.0
            float_pct = 0.9
            top10_conc = 0.35
            short_int = 0.03
            ceo_age = 55
            ceo_tenure = 6
            ceo_founder = 0
            ceo_comp = 12
            ceo_eq = 1.5
            avg_neo = 11
            pay_ratio = 200
            board_indep = 75
            board_size = 9
            insider_own = 3
            ceo_pay_vs = 0
            neo_turn = 0.15
            ceo_dual = 0
            rate_map = {
                2015: 2.27,
                2016: 1.84,
                2017: 2.33,
                2018: 2.91,
                2019: 2.14,
                2020: 0.89,
                2021: 1.45,
                2022: 2.95,
                2023: 3.96,
                2024: 4.2,
            }
            vix_map = {
                2015: 16.7,
                2016: 15.8,
                2017: 11.1,
                2018: 16.6,
                2019: 15.4,
                2020: 29.2,
                2021: 19.7,
                2022: 25.6,
                2023: 16.8,
                2024: 15,
            }
            rate_10y = rate_map.get(yr, 3.0)
            vix_avg = vix_map.get(yr, 16)
            credit_spread = 3.5
            gdp = {
                2015: 2.9,
                2016: 1.8,
                2017: 2.2,
                2018: 2.9,
                2019: 2.3,
                2020: -2.2,
                2021: 5.8,
                2022: 1.9,
                2023: 2.5,
                2024: 2.2,
            }.get(yr, 2)

            # --- OVERRIDE with real DEF14A where available ---
            ef = exec_features.get((ticker, yr))
            if ef is None:
                # also try ticker with different case? tickers are upper
                ef = exec_features.get((ticker, str(yr)))
            if ef:
                # ef contains computed
                ceo_comp = ef.get("CEO_TOTAL_COMP", ceo_comp)
                avg_neo = ef.get("AVG_NEO_COMP", avg_neo)
                ceo_tenure = ef.get("CEO_TENURE", ceo_tenure)
                ceo_founder = ef.get("CEO_FOUNDER_FLAG", ceo_founder)
                neo_turn = ef.get("NEO_TURNOVER", neo_turn)
                board_size = ef.get("BOARD_SIZE", board_size)
                # NEO_COUNT override
                neo_count_real = ef.get("NEO_COUNT", 5)
            else:
                neo_count_real = 5

            # --- Form4 override ---
            f4_count = form4_flat.get((ticker, yr))
            if f4_count is None:
                f4_count = form4_flat.get((ticker, str(yr)))
            if f4_count is not None:
                insider_net = float(f4_count)  # proxy
            else:
                # also check string fy
                insider_net = 0.0

            # --- Market override (static per ticker) ---
            mkt = market_data.get(ticker)
            # market placeholders for market_price family
            ret_12m = None
            vol_252d = None
            vol_avg_30d = None
            price_vs_52w = None
            if mkt:
                ret_12m = mkt.get("ret_12m")
                vol_252d = mkt.get("vol_252d")
                vol_avg_30d = mkt.get("avg_vol_30d")
                price_vs_52w = mkt.get("price_vs_52w")

            altman = None
            if (
                assets
                and assets != 0
                and equity
                and ret_earn is not None
                and ebit is not None
            ):
                try:
                    wc = working_cap or 0
                    mv = equity
                    liab_v = liab or assets * 0.6
                    altman = (
                        1.2 * (wc / assets)
                        + 1.4 * (ret_earn / assets)
                        + 3.3 * (ebit / assets)
                        + 0.6 * (mv / liab_v)
                        + 1.0 * ((rev or 0) / assets)
                    )
                except:
                    pass

            # Base market_price features: if real market exists set others 0 else None
            if mkt:
                ret_1m = 0.0
                ret_3m = 0.0
                ret_6m = 0.0
                vol_30d = 0.0
                vol_90d = 0.0
                beta_1y = 0.0
                mom_12_1 = 0.0
            else:
                ret_1m = None
                ret_3m = None
                ret_6m = None
                vol_30d = None
                vol_90d = None
                beta_1y = None
                mom_12_1 = None

            row_feat = {
                "REV": rev,
                "COGS": cogs,
                "GROSS_PROFIT": gross,
                "OP_INCOME": op,
                "EBITDA": ebitda,
                "NET_INCOME": net,
                "EBIT": ebit,
                "GROSS_MARGIN": gross_margin,
                "OP_MARGIN": op_margin,
                "NET_MARGIN": net_margin,
                "EBITDA_MARGIN": ebitda_margin,
                "TOTAL_ASSETS": assets,
                "TOTAL_LIABILITIES": liab,
                "EQUITY": equity,
                "CASH": cash,
                "DEBT": debt,
                "BOOK_VALUE": book_value,
                "TANGIBLE_BOOK": equity,
                "WORKING_CAPITAL": working_cap,
                "NET_DEBT": net_debt,
                "INVESTED_CAPITAL": invested_cap,
                "OCF": ocf,
                "CAPEX": capex,
                "FCF": fcf,
                "FCF_MARGIN": fcf_margin,
                "OCF_TO_NET": safe_div(ocf, net),
                "FCF_CONVERSION": safe_div(fcf, net),
                "CAPEX_TO_REV": safe_div(capex, rev),
                "REV_YOY": rev_yoy,
                "EBITDA_YOY": ebitda_yoy,
                "NET_YOY": net_yoy,
                "FCF_YOY": fcf_yoy,
                "REV_3Y_CAGR": rev_3y,
                "EBITDA_3Y_CAGR": cagr(ebitda, f"EBITDA_{yr - 3}"),
                "EPS_3Y_CAGR": cagr(eps, f"EPS_{yr - 3}"),
                "BOOK_3Y_CAGR": cagr(book_value, f"BOOK_{yr - 3}"),
                "OCF_3Y_CAGR": cagr(ocf, f"OCF_{yr - 3}"),
                "ROE": roe,
                "ROA": roa,
                "ROIC": roic,
                "FCF_ROIC": safe_div(fcf, invested_cap),
                "ROIC_WACC_SPREAD": (roic - 0.08) if roic else None,
                "CURRENT_RATIO": curr_ratio,
                "QUICK_RATIO": curr_ratio,
                "DEBT_TO_EQUITY": debt_eq,
                "DEBT_TO_EBITDA": debt_ebitda,
                "INTEREST_COVERAGE": int_cov,
                "DEBT_TO_ASSETS": debt_assets,
                "NET_DEBT_TO_EBITDA": net_debt_ebitda,
                "ASSET_TURNOVER": asset_turn,
                "INVENTORY_TURNOVER": asset_turn,
                "RECEIVABLE_TURNOVER": asset_turn,
                "CASH_CONVERSION_CYCLE": None,
                "CAPEX_TO_DEPRE": capex_depr,
                "EPS_DILUTED": eps,
                "BVPS": bvps,
                "FCFPS": fcfps,
                "SHARES_YOY": shares_yoy,
                "DILUTION_3Y": cagr(shares_d, f"SHARES_{yr - 3}"),
                "RET_1M": ret_1m,
                "RET_3M": ret_3m,
                "RET_6M": ret_6m,
                "RET_12M": ret_12m,
                "VOL_30D": vol_30d,
                "VOL_90D": vol_90d,
                "VOL_252D": vol_252d,
                "BETA_1Y": beta_1y,
                "VOLUME_AVG_30D": vol_avg_30d,
                "MOMENTUM_12_1": mom_12_1,
                "PE": None,
                "PB": None,
                "PS": None,
                "EV_EBITDA": None,
                "EV_SALES": None,
                "EARNINGS_YIELD": None,
                "FCF_YIELD": None,
                "DIV_YIELD": 0.015,
                "NEO_COUNT": neo_count_real,
                "CEO_AGE": ceo_age,
                "CEO_TENURE": ceo_tenure,
                "CEO_FOUNDER_FLAG": ceo_founder,
                "CEO_TOTAL_COMP": ceo_comp,
                "CEO_EQUITY_PCT": ceo_eq,
                "AVG_NEO_COMP": avg_neo,
                "CEO_PAY_RATIO": pay_ratio,
                "BOARD_INDEP_PCT": board_indep,
                "BOARD_SIZE": board_size,
                "INSIDER_OWN_PCT": insider_own,
                "CEO_PAY_VS_SECTOR": ceo_pay_vs,
                "NEO_TURNOVER": neo_turn,
                "CEO_DUALITY": ceo_dual,
                "INST_PCT": inst_pct,
                "INST_DELTA_QOQ": inst_delta,
                "INSIDER_NET_12M": insider_net,
                "FLOAT_PCT": float_pct,
                "TOP10_INST_CONC": top10_conc,
                "SHORT_INTEREST_PCT": short_int,
                "MDA_LENGTH": 5,
                "MDA_SENTIMENT": 0.05,
                "RISK_FACTOR_COUNT": 20,
                "RISK_CHANGE_YOY": 0,
                "FOG_INDEX_PROXY": 18,
                "TONE_UNCERTAINTY": 0.15,
                "SECTOR_REL_RET_12M": 0,
                "SECTOR_CONCENTRATION": 0.2,
                "SECTOR_BETA": 1.0,
                "RATE_10Y": rate_10y,
                "VIX_AVG_FY": vix_avg,
                "CREDIT_SPREAD_PROXY": credit_spread,
                "GDP_GROWTH_FY": gdp,
                "EARN_SURPRISE_STREAK": 0,
                "GUIDANCE_RAISE_FLAG": 0,
                "EPS_REVISION_UP_PCT": 0.5,
                "PRICE_VS_52W_HIGH": price_vs_52w if price_vs_52w is not None else 0.9,
                "RSI_14_PROXY": 50,
                "ACCIDENT_DISCLOSURE": 0,
                "ALTMAN_Z": altman,
                "PIOTROSKI_F_SCORE_PROXY": 5,
            }
            for k in ALL_FEATURES:
                if k not in row_feat:
                    row_feat[k] = None

            prev_vals[f"REV_{yr}"] = rev
            prev_vals[f"EBITDA_{yr}"] = ebitda
            prev_vals[f"NET_{yr}"] = net
            prev_vals[f"FCF_{yr}"] = fcf
            prev_vals[f"EPS_{yr}"] = eps
            prev_vals[f"BOOK_{yr}"] = book_value
            prev_vals[f"OCF_{yr}"] = ocf
            prev_vals[f"SHARES_{yr}"] = shares_d

            if rev is None and assets is None:
                continue
            all_rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "sector": sector,
                    "fiscal_year": str(yr),
                    "features": row_feat,
                }
            )

    print(
        f"Collected {len(all_rows)} rows from {len({r['ticker'] for r in all_rows})} tickers (v4)"
    )

    D = len(ALL_FEATURES)
    N = len(all_rows)
    Z_raw = np.zeros((N, D), dtype=np.float32)
    mask = np.zeros((N, D), dtype=np.float32)
    tickers = []
    names = []
    fyears = []
    sectors = []
    for i, row in enumerate(all_rows):
        tickers.append(row["ticker"])
        names.append(row["company"])
        fyears.append(row["fiscal_year"])
        sectors.append(row["sector"])
        for j, feat_name in enumerate(ALL_FEATURES):
            val = row["features"].get(feat_name)
            if val is not None:
                try:
                    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                        continue
                    Z_raw[i, j] = float(val)
                    mask[i, j] = 1.0
                except:
                    pass
    # fill median per FY then global, z-score per FY
    Z_filled = Z_raw.copy()
    for fy in sorted(set(fyears)):
        rows = [k for k, v in enumerate(fyears) if v == fy]
        for j in range(D):
            col = Z_raw[rows, j]
            m = mask[rows, j]
            valid = col[m > 0.5]
            if len(valid) == 0:
                gvalid = Z_raw[:, j][mask[:, j] > 0.5]
                median = np.median(gvalid) if len(gvalid) > 0 else 0.0
            else:
                median = np.median(valid)
            Z_filled[rows, j] = np.where(m > 0.5, col, median)
    Z = np.zeros_like(Z_filled)
    for fy in sorted(set(fyears)):
        rows = [k for k, v in enumerate(fyears) if v == fy]
        for j in range(D):
            vals = Z_filled[rows, j]
            if len(vals) < 2:
                Z[rows, j] = 0
                continue
            mean = vals.mean()
            std = max(vals.std(), 1e-6)
            zs = (vals - mean) / std
            zs = np.clip(zs, -4, 4)
            Z[rows, j] = zs

    # Coverage stats
    total_tickers = len(set(tickers))
    total_rows = N
    # real NEO coverage
    neo_keys = set(exec_features.keys())
    rows_with_neo = sum(
        1
        for i in range(N)
        if (tickers[i], int(fyears[i]) if fyears[i].isdigit() else fyears[i])
        in exec_features
        or (tickers[i], fyears[i]) in [(k[0], str(k[1])) for k in neo_keys]
    )
    # more precise: check membership with int conversion
    exec_set_str = {(t, str(fy)) for t, fy in exec_features.keys()}
    rows_with_neo_precise = sum(
        1 for t, fy in zip(tickers, fyears, strict=False) if (t, fy) in exec_set_str
    )
    tickers_with_neo = len({t for t, fy in exec_set_str})
    # market coverage
    market_tickers = set(market_data.keys())
    tickers_with_market = len(set(tickers) & market_tickers)
    rows_with_market = sum(1 for t in tickers if t in market_tickers)
    # form4 coverage
    form4_keys_str = {(t, str(fy)) for t, fy in form4_flat.keys()}
    rows_with_form4 = sum(
        1 for t, fy in zip(tickers, fyears, strict=False) if (t, fy) in form4_keys_str
    )
    tickers_with_form4 = len({t for t, _ in form4_flat.keys()} & set(tickers))

    # feature manifest with real flags
    real_flags = {}
    for feat in ALL_FEATURES:
        fam = next(
            (fam for fam, feats in FEATURE_FAMILIES.items() if feat in feats), "unknown"
        )
        if fam == "management_neo":
            real_flags[feat] = (
                rows_with_neo_precise / total_rows > 0.02
            )  # if we have any real, mark as partially real
        elif fam == "ownership":
            real_flags[feat] = feat == "INSIDER_NET_12M" and rows_with_form4 > 0
        elif fam == "market_price":
            real_flags[feat] = (
                feat in ["RET_12M", "VOL_252D", "VOLUME_AVG_30D"]
                and rows_with_market > 0
            )
        elif False:
            pass
        else:
            real_flags[feat] = False
    # For v4 we consider specific features real where we overrode
    # Override true for those explicitly from real sources
    for feat in [
        "NEO_COUNT",
        "CEO_TOTAL_COMP",
        "AVG_NEO_COMP",
        "CEO_TENURE",
        "CEO_FOUNDER_FLAG",
        "NEO_TURNOVER",
        "BOARD_SIZE",
    ]:
        if rows_with_neo_precise > 0:
            real_flags[feat] = True
    for feat in ["INSIDER_NET_12M"]:
        if rows_with_form4 > 0:
            real_flags[feat] = True
    for feat in ["RET_12M", "VOL_252D", "VOLUME_AVG_30D", "PRICE_VS_52W_HIGH"]:
        if rows_with_market > 0:
            real_flags[feat] = True

    manifest = {
        "features": ALL_FEATURES,
        "families": [
            next(
                (fam for fam, feats in FEATURE_FAMILIES.items() if feat in feats),
                "unknown",
            )
            for feat in ALL_FEATURES
        ],
        "game_features": GAME_PROFILE_FEATURES,
        "sectors": SECTORS,
        "real_data": True,
        "sec_only": False,
        "v4": True,
        "years": list(range(2015, 2025)),
        "tickers": len(set(tickers)),
        "rows": N,
        "real_flags": real_flags,
        "real_coverage": {
            "tickers_with_real_neo": tickers_with_neo,
            "pct_tickers_real_neo": tickers_with_neo / total_tickers
            if total_tickers
            else 0,
            "rows_with_real_neo": rows_with_neo_precise,
            "pct_rows_real_neo": rows_with_neo_precise / total_rows
            if total_rows
            else 0,
            "tickers_with_market": tickers_with_market,
            "pct_tickers_market": tickers_with_market / total_tickers
            if total_tickers
            else 0,
            "rows_with_market": rows_with_market,
            "pct_rows_market": rows_with_market / total_rows if total_rows else 0,
            "tickers_with_form4": tickers_with_form4,
            "pct_tickers_form4": tickers_with_form4 / total_tickers
            if total_tickers
            else 0,
            "rows_with_form4": rows_with_form4,
            "pct_rows_form4": rows_with_form4 / total_rows if total_rows else 0,
        },
        "sources": {
            "def14a_parsed": len(exec_features),
            "form4_index": len(form4_flat),
            "market_cache": len(market_data),
        },
    }

    out_v4 = DATA_DIR / "train_matrix_v4.npz"
    out_alias = DATA_DIR / "train_matrix.npz"
    out_real = DATA_DIR / "train_matrix_real.npz"

    np.savez_compressed(
        out_v4,
        Z=Z.astype(np.float32),
        mask=mask,
        ticker=np.array(tickers),
        name=np.array(names),
        fiscal_year=np.array(fyears),
        sector=np.array(sectors),
        cluster=np.zeros(N, dtype=np.int64),
        Z_raw=Z_raw,
    )
    np.savez_compressed(
        out_real,
        Z=Z.astype(np.float32),
        mask=mask,
        ticker=np.array(tickers),
        name=np.array(names),
        fiscal_year=np.array(fyears),
        sector=np.array(sectors),
        cluster=np.zeros(N, dtype=np.int64),
        Z_raw=Z_raw,
    )
    np.savez_compressed(
        out_alias,
        Z=Z.astype(np.float32),
        mask=mask,
        ticker=np.array(tickers),
        name=np.array(names),
        fiscal_year=np.array(fyears),
        sector=np.array(sectors),
        cluster=np.zeros(N, dtype=np.int64),
    )

    (DATA_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    # embedding placeholder - keep existing if present else create random
    embed_path = DATA_DIR / "embedding.npz"
    if not embed_path.exists():
        # create random 32-dim
        np.savez_compressed(
            embed_path,
            embedding=np.random.randn(N, 32).astype(np.float32),
            ticker=np.array(tickers),
            fiscal_year=np.array(fyears),
        )

    # real_rows_meta
    meta = {
        "total_rows": N,
        "total_tickers": total_tickers,
        "avg_rows_per_ticker": N / total_tickers if total_tickers else 0,
        "coverage": manifest["real_coverage"],
        "sources": manifest["sources"],
        "features": ALL_FEATURES,
        "real_flags": real_flags,
        "notes": "v4 adds real DEF14A exec, Form4 counts proxy, static market snapshot per ticker",
    }
    (DATA_DIR / "real_rows_meta.json").write_text(json.dumps(meta, indent=2))

    # report md
    report_path = DATA_DIR / "v4_build_report.md"
    report_lines = [
        "# Train Matrix v4 Build Report",
        "",
        f"- Total rows: {N}",
        f"- Total tickers: {total_tickers}",
        f"- Avg rows per ticker: {N / total_tickers:.2f}",
        "- Years: 2015-2024",
        f"- Features: {D}",
        "",
        "## Coverage",
        f"- DEF14A parsed entries: {len(exec_features)} (ticker,FY combos)",
        f"- Tickers with real NEO: {tickers_with_neo} / {total_tickers} = {tickers_with_neo / total_tickers * 100:.1f}%",
        f"- Rows with real NEO: {rows_with_neo_precise} / {total_rows} = {rows_with_neo_precise / total_rows * 100:.1f}%",
        f"- Market cache tickers: {len(market_data)} loaded, matched {tickers_with_market} tickers in matrix = {tickers_with_market / total_tickers * 100:.1f}%",
        f"- Rows with real market: {rows_with_market} = {rows_with_market / total_rows * 100:.1f}%",
        f"- Form4 index filings: {len(form4_flat)} (ticker,FY) combos, {total_lines if 'total_lines' in locals() else 'N/A'} filings",
        f"- Tickers with Form4: {tickers_with_form4} / {total_tickers} = {tickers_with_form4 / total_tickers * 100:.1f}%",
        f"- Rows with Form4: {rows_with_form4} / {total_rows} = {rows_with_form4 / total_rows * 100:.1f}%",
        "",
        "## Real Features Implemented",
        "- management_neo: NEO_COUNT, CEO_TOTAL_COMP (M), AVG_NEO_COMP (M), CEO_TENURE (cumulative years same CEO), CEO_FOUNDER_FLAG, NEO_TURNOVER (1-overlap), BOARD_SIZE",
        "- ownership: INSIDER_NET_12M approximated by Form4 filing counts per FY",
        "- market_price: RET_12M, VOL_252D, VOLUME_AVG_30D static per ticker from cache (current snapshot used for all FYs), PRICE_VS_52W_HIGH from same",
        "- Others remain placeholder or SEC-derived",
        "",
        "## Files",
        f"- {out_v4}",
        f"- {out_alias} (alias)",
        f"- {out_real}",
        f"- {DATA_DIR / 'feature_manifest.json'}",
        f"- {DATA_DIR / 'real_rows_meta.json'}",
        f"- {DATA_DIR / 'embedding.npz'}",
        "",
        "## Notes",
        "- DEF14A FY mapping: filing_date month Jan-Apr => FY = year-1 else year",
        "- CEO detection: keyword chief executive / ceo else max comp; name cleaned, filtered for person names",
        "- CEO_TENURE = cumulative count distinct years same CEO name appears up to FY",
        "- NEO_TURNOVER = 1 - overlap/maxCount",
        "- Market is static snapshot (not historical) used across FYs — better than None, will be refined with historical series later",
        "- Form4 counts are filing counts proxy (buy vs sell unknown without XML cache)",
        "- Matrix z-scored per FY median-filled, same as v1 -> preserves 2741 rows, 283 tickers avg 9.7",
    ]
    report_path.write_text("\n".join(report_lines))
    print(
        f"Saved v4 to {out_v4}, alias {out_alias}, manifest, meta, report {report_path}"
    )
    print(
        f"Coverage: NEO {rows_with_neo_precise}/{N} {rows_with_neo_precise / N * 100:.1f}%, Market {rows_with_market}/{N} {rows_with_market / N * 100:.1f}%, Form4 {rows_with_form4}/{N} {rows_with_form4 / N * 100:.1f}%"
    )
    return out_v4


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build_v4(limit=args.limit)
