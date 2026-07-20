"""
Build real matrix v5 — continuous career redesign
- Uses market_history 10y daily per ticker for per-FY historical market metrics + forward returns + triple barrier
- Computes valuation historically per FY from price at FY-end
- Detects CEO change events from DEF14A timeline
- Outputs train_matrix_v5.npz with Z, mask, plus forward labels

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
MARKET_HIST_DIR = ROOT / "pipeline" / "cache" / "market_history"

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
    if "$" in cn or re.search(r"\d{4,}", cn):
        return False
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
            return False
    parts = cn.split()
    if len(parts) < 2:
        return False
    if not (parts[0][0].isupper() and parts[1][0].isupper()):
        return False
    if parts[-1].lower() == "and":
        return False
    if low.startswith("chairman of the boar") or low.startswith("chairman of"):
        return False
    if low.startswith("chief executive"):
        return False
    if len(cn) > 35 and any(
        k in low for k in ["officer", "chairman", "president", "vice"]
    ):
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
        except:
            pass
    return out


def load_def14a():
    candidates = [
        DATA_DIR / "def14a_parsed_v3.jsonl",
        DATA_DIR / "def14a_parsed_v2.jsonl",
        DATA_DIR / "def14a_parsed.jsonl",
    ]
    path = None
    for c in candidates:
        if c.exists() and c.stat().st_size > 100:
            path = c
            break
    if path is None:
        print("No DEF14A parsed found")
        return {}
    print(f"Loading DEF14A from {path} size {path.stat().st_size}")
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
    tmp = {}
    for entry in raw_entries:
        ticker = entry.get("ticker")
        fdate_str = entry.get("filing_date", "")
        try:
            yr = int(fdate_str[:4])
            mo = int(fdate_str[5:7]) if len(fdate_str) >= 7 else 6
        except:
            continue
        if mo >= 1 and mo <= 4:
            fy = yr - 1
        else:
            fy = yr
        key = (ticker, fy)
        existing = tmp.get(key)
        if existing is None or fdate_str > existing.get("filing_date", ""):
            tmp[key] = entry
    by_ticker = defaultdict(list)
    for (ticker, fy), ent in tmp.items():
        by_ticker[ticker].append((fy, ent))
    exec_features = {}
    for ticker, lst in by_ticker.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        ceo_counts = Counter()
        prev_neo_names = None
        prev_ceo_name = None
        for fy, ent in lst_sorted:
            neos_raw = ent.get("neos", [])
            board_size = ent.get("board_size")
            neos_filtered = []
            for neo in neos_raw:
                n_name = neo.get("name", "")
                if is_person_name(n_name):
                    neos_filtered.append(neo)
                else:
                    cn = clean_name(n_name)
                    parts = cn.split()
                    if (
                        len(parts) >= 2
                        and parts[0][0].isupper()
                        and parts[1][0].isupper()
                    ):
                        tentative = (
                            " ".join(parts[:3])
                            if len(parts) >= 3
                            else " ".join(parts[:2])
                        )
                        if is_person_name(tentative):
                            neo_copy = dict(neo)
                            neo_copy["name"] = tentative
                            neos_filtered.append(neo_copy)
            if not neos_filtered and neos_raw:
                neos_filtered = neos_raw
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
            if ceo_candidate is None and neos_filtered:
                ceo_candidate = max(
                    neos_filtered, key=lambda x: x.get("total_comp", 0) or 0
                )
            if ceo_candidate is None:
                continue
            ceo_name_raw = ceo_candidate.get("name", "")
            ceo_name = clean_name(ceo_name_raw)
            ceo_name = re.sub(r"\s+Chairman.*$", "", ceo_name, flags=re.I).strip()
            ceo_name = re.sub(r"\s+Chief.*$", "", ceo_name, flags=re.I).strip()
            ceo_name = clean_name(ceo_name)

            def comp_million(c):
                if c is None:
                    return None
                try:
                    v = float(c)
                    if v > 1000:
                        return v / 1e6
                    else:
                        return v
                except:
                    return None

            ceo_comp = comp_million(ceo_candidate.get("total_comp"))
            comps = [comp_million(n.get("total_comp")) for n in neos_filtered]
            comps = [c for c in comps if c is not None]
            avg_comp = float(np.mean(comps)) if comps else None
            neo_count = len(neos_filtered) if neos_filtered else ent.get("neo_count", 0)
            combined_ceo = (
                ceo_candidate.get("name", "") + " " + ceo_candidate.get("row_text", "")
            ).lower()
            founder_flag = 1 if "founder" in combined_ceo else 0
            bs = (
                board_size
                if isinstance(board_size, int) and 3 <= board_size <= 20
                else 9
            )
            ceo_counts[ceo_name.lower()] += 1
            tenure = ceo_counts[ceo_name.lower()]
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
                (prev_ceo_name is not None and ceo_name.lower() != prev_ceo_name)
            # time since CEO change will be computed later per sequence
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
                "CEO_CHANGE_FLAG": 1
                if (prev_ceo_name is not None and ceo_name.lower() != prev_ceo_name)
                else 0,
            }
            prev_neo_names = curr_neo_names
            prev_ceo_name = ceo_name.lower()
    print(f"Computed exec features for {len(exec_features)} (ticker,FY) combos")
    return exec_features


def load_form4():
    candidates = [
        DATA_DIR / "form4_index_all.jsonl",
        DATA_DIR / "form4_index.jsonl",
    ]
    files = [c for c in candidates if c.exists()]
    if not files:
        print("No Form4 index found")
        return {}
    counts = defaultdict(lambda: defaultdict(int))
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
                    counts[ticker][yr] += 1
                    total_lines += 1
                except:
                    continue
    flat = {}
    for ticker, year_dict in counts.items():
        for fy, cnt in year_dict.items():
            flat[(ticker, fy)] = cnt
    print(f"Form4 total filings {total_lines}, tickers {len(counts)}")
    return flat


def load_market_history():
    out = {}
    if not MARKET_HIST_DIR.exists():
        print("Market history dir missing, will fallback")
        return out
    for fp in MARKET_HIST_DIR.glob("*.json"):
        try:
            j = json.loads(fp.read_text())
            ticker = j.get("ticker")
            if not ticker:
                continue
            # keep history as list of dicts sorted by date
            # ensure sorted
            hist = j.get("history", [])
            hist_sorted = sorted(hist, key=lambda x: x["date"])
            j["history"] = hist_sorted
            out[ticker] = j
        except Exception as e:
            print(f"fail load {fp}: {e}")
    print(f"Loaded market history for {len(out)} tickers (10y daily)")
    return out


def compute_market_metrics_at_fy(history, spy_history, fy):
    """
    history: list of {date, close, ...} sorted asc
    spy_history optional for beta
    fy: int year (e.g., 2024) -> FY-end = Dec 31 of fy
    Returns dict of market features at that date + forward labels if future data exists
    """
    # target date
    target_str = f"{fy}-12-31"
    # find idx of last trading day <= target_str
    # binary search
    import bisect

    dates = [h["date"] for h in history]
    idx = bisect.bisect_right(dates, target_str) - 1
    if idx < 0:
        return None
    if idx < 252:
        # not enough history for 12M metrics? Still compute with available but mark weak
        pass

    def get_close(i):
        return history[i]["close"] if 0 <= i < len(history) else None

    def ret_between(i_now, days_back):
        c_now = get_close(i_now)
        c_then = get_close(i_now - days_back)
        if c_now is None or c_then is None or c_then == 0:
            return None
        return c_now / c_then - 1

    c_end = get_close(idx)
    if c_end is None:
        return None
    # daily returns last 252 for vol, beta
    closes = np.array([h["close"] for h in history], dtype=float)
    rets = np.zeros_like(closes)
    rets[1:] = (closes[1:] - closes[:-1]) / closes[:-1]

    # vol
    def vol_for(window, end_idx):
        if end_idx - window < 0:
            return None
        window_rets = rets[end_idx - window + 1 : end_idx + 1]
        if len(window_rets) < 5:
            return None
        # std of daily * sqrt(252)
        return float(np.std(window_rets) * np.sqrt(252))

    ret_1m = ret_between(idx, 21)
    ret_3m = ret_between(idx, 63)
    ret_6m = ret_between(idx, 126)
    ret_12m = ret_between(idx, 252)
    vol_30 = vol_for(30, idx)
    vol_90 = vol_for(90, idx)
    vol_252 = vol_for(252, idx)
    # volume avg 30d
    vols = np.array([h["volume"] for h in history], dtype=float)
    vol_avg_30 = float(np.mean(vols[max(0, idx - 29) : idx + 1])) if idx >= 0 else None
    # price vs 52w high
    window_252_closes = closes[max(0, idx - 251) : idx + 1]
    high_52 = max(window_252_closes) if len(window_252_closes) > 0 else None
    price_vs_52w = c_end / high_52 if high_52 and high_52 != 0 else None
    # momentum 12_1
    mom_12_1 = None
    if ret_12m is not None and ret_1m is not None:
        mom_12_1 = ret_12m - ret_1m
    # RSI 14 proxy
    rsi = None
    if idx >= 14:
        gains = []
        losses = []
        for k in range(idx - 13, idx + 1):
            if k <= 0:
                continue
            diff = closes[k] - closes[k - 1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-diff)
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 1e-6
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    # beta vs SPY
    beta = None
    if spy_history is not None and len(spy_history) > 252:
        try:
            spy_list = (
                spy_history
                if isinstance(spy_history, list)
                else spy_history.get("history", [])
            )
            spy_closes = np.array([h["close"] for h in spy_list], dtype=float)
            spy_dates = [h["date"] for h in spy_list]
            spy_idx = bisect.bisect_right(spy_dates, dates[idx]) - 1
            if spy_idx >= 252:

                def spy_rets_at(s_idx, window):
                    sc = spy_closes
                    sr = np.zeros_like(sc)
                    sr[1:] = (sc[1:] - sc[:-1]) / sc[:-1]
                    return sr[s_idx - window + 1 : s_idx + 1]

                stock_window = rets[max(1, idx - 251) : idx + 1]
                spy_window = spy_rets_at(spy_idx, len(stock_window))
                if len(stock_window) == len(spy_window) and len(stock_window) >= 50:
                    cov = np.cov(stock_window, spy_window)[0, 1]
                    var_spy = np.var(spy_window)
                    if var_spy > 1e-8:
                        beta = cov / var_spy
        except Exception:
            beta = None
    # forward labels
    fwd_ret_1m = None
    fwd_ret_3m = None
    fwd_ret_6m = None
    fwd_ret_12m = None
    fwd_vol_6m = None
    fwd_dd_6m = None
    triple_barrier = None
    # forward needs future data
    if idx + 252 < len(history):

        def fwd_ret(days):
            cf = get_close(idx + days)
            if cf is None:
                return None
            return cf / c_end - 1

        fwd_ret_1m = fwd_ret(21)
        fwd_ret_3m = fwd_ret(63)
        fwd_ret_6m = fwd_ret(126)
        fwd_ret_12m = fwd_ret(252)
        # forward vol 6M
        fwd_vol_6m = vol_for(126, idx + 126) if idx + 126 < len(history) else None
        # max drawdown next 6M
        future_closes = (
            closes[idx + 1 : idx + 127]
            if idx + 127 < len(closes)
            else closes[idx + 1 :]
        )
        if len(future_closes) > 5:
            peak = np.maximum.accumulate(future_closes)
            drawdown = (future_closes - peak) / peak
            fwd_dd_6m = float(np.min(drawdown)) if len(drawdown) > 0 else None
        # triple barrier +10% profit, -7% loss within 63 days
        profit_level = c_end * 1.10
        loss_level = c_end * 0.93
        barrier_hit = None
        for j in range(1, min(64, len(closes) - idx)):
            price = closes[idx + j]
            if price >= profit_level:
                barrier_hit = 1
                break
            if price <= loss_level:
                barrier_hit = 0
                break
        if barrier_hit is None:
            barrier_hit = 0  # no profit within window = not entry
        triple_barrier = barrier_hit
    return {
        "price": c_end,
        "RET_1M": ret_1m,
        "RET_3M": ret_3m,
        "RET_6M": ret_6m,
        "RET_12M": ret_12m,
        "VOL_30D": vol_30,
        "VOL_90D": vol_90,
        "VOL_252D": vol_252,
        "BETA_1Y": beta,
        "VOLUME_AVG_30D": vol_avg_30,
        "MOMENTUM_12_1": mom_12_1,
        "PRICE_VS_52W_HIGH": price_vs_52w,
        "RSI_14_PROXY": rsi,
        # forward
        "FWD_RET_1M": fwd_ret_1m,
        "FWD_RET_3M": fwd_ret_3m,
        "FWD_RET_6M": fwd_ret_6m,
        "FWD_RET_12M": fwd_ret_12m,
        "FWD_VOL_6M": fwd_vol_6m,
        "FWD_DD_6M": fwd_dd_6m,
        "TRIPLE_BARRIER": triple_barrier,
    }


def build_v5(limit=None):
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} summaries, universe {len(uni)}")
    exec_features = load_def14a()
    form4_flat = load_form4()
    market_hist = load_market_history()
    spy_hist = market_hist.get("SPY")

    # fallback static market
    market_static = {}
    if (ROOT / "pipeline" / "cache" / "market").exists():
        for fp in (ROOT / "pipeline" / "cache" / "market").glob("*.json"):
            try:
                j = json.loads(fp.read_text())
                t = j.get("ticker")
                if t:
                    market_static[t] = j
            except:
                pass

    all_rows = []
    fwd_labels = []  # parallel list of dicts per row
    ceo_history_per_ticker = defaultdict(list)  # ticker -> list of (fy, ceo_name)

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
        # pre-load CEO names for tenure re-calc continuous
        # exec_features already has tenure but we will also compute time since change
        for yr in range(2014, 2025):
            ef = exec_features.get((ticker, yr))
            if ef and "CEO_NAME" in ef:
                ceo_history_per_ticker[ticker].append((yr, ef["CEO_NAME"]))
        # sort
        ceo_history_per_ticker[ticker] = sorted(ceo_history_per_ticker[ticker])

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
            ceo_change_flag = 0
            time_since_ceo_change = 0

            ef = exec_features.get((ticker, yr))
            if ef:
                ceo_comp = ef.get("CEO_TOTAL_COMP", ceo_comp)
                avg_neo = ef.get("AVG_NEO_COMP", avg_neo)
                ceo_tenure = ef.get("CEO_TENURE", ceo_tenure)
                ceo_founder = ef.get("CEO_FOUNDER_FLAG", ceo_founder)
                neo_turn = ef.get("NEO_TURNOVER", neo_turn)
                board_size = ef.get("BOARD_SIZE", board_size)
                neo_count_real = ef.get("NEO_COUNT", 5)
                ceo_change_flag = ef.get("CEO_CHANGE_FLAG", 0)
                # time since change: walk back
                # find last change year
                last_change_yr = None
                # we need previous changes list; we have ceo_history_per_ticker but includes current
                # compute by scanning sorted history
                sorted_ceo = sorted(
                    [
                        (fy_, nm)
                        for fy_, nm in ceo_history_per_ticker[ticker]
                        if fy_ <= yr
                    ],
                    key=lambda x: x[0],
                )
                for i in range(1, len(sorted_ceo)):
                    if sorted_ceo[i][1].lower() != sorted_ceo[i - 1][1].lower():
                        last_change_yr = sorted_ceo[i][0]
                if last_change_yr is None:
                    # no change recorded
                    time_since_ceo_change = ceo_tenure
                else:
                    time_since_ceo_change = yr - last_change_yr
            else:
                neo_count_real = 5

            f4_count = form4_flat.get((ticker, yr))
            if f4_count is not None:
                insider_net = float(f4_count)

            # Market history metrics
            m_hist = market_hist.get(ticker)
            m_metrics = None
            if m_hist and m_hist.get("history"):
                spy_list = spy_hist.get("history") if spy_hist else None
                m_metrics = compute_market_metrics_at_fy(
                    m_hist.get("history"), spy_list, yr
                )
            # fallback static
            if m_metrics is None:
                mkt_static = market_static.get(ticker)
                if mkt_static:
                    # use static as before
                    m_metrics = {
                        "price": mkt_static.get("last_close"),
                        "RET_1M": 0.0,
                        "RET_3M": 0.0,
                        "RET_6M": 0.0,
                        "RET_12M": mkt_static.get("ret_12m"),
                        "VOL_30D": 0.0,
                        "VOL_90D": 0.0,
                        "VOL_252D": mkt_static.get("vol_252d"),
                        "BETA_1Y": 0.0,
                        "VOLUME_AVG_30D": mkt_static.get("avg_vol_30d"),
                        "MOMENTUM_12_1": 0.0,
                        "PRICE_VS_52W_HIGH": mkt_static.get("price_vs_52w"),
                        "RSI_14_PROXY": 50,
                        "FWD_RET_1M": None,
                        "FWD_RET_3M": None,
                        "FWD_RET_6M": None,
                        "FWD_RET_12M": None,
                        "FWD_VOL_6M": None,
                        "FWD_DD_6M": None,
                        "TRIPLE_BARRIER": None,
                    }
            # extract
            price_at_fy = m_metrics["price"] if m_metrics else None
            ret_1m = m_metrics.get("RET_1M") if m_metrics else None
            ret_3m = m_metrics.get("RET_3M") if m_metrics else None
            ret_6m = m_metrics.get("RET_6M") if m_metrics else None
            ret_12m = m_metrics.get("RET_12M") if m_metrics else None
            vol_30d = m_metrics.get("VOL_30D") if m_metrics else None
            vol_90d = m_metrics.get("VOL_90D") if m_metrics else None
            vol_252d = m_metrics.get("VOL_252D") if m_metrics else None
            beta_1y = m_metrics.get("BETA_1Y") if m_metrics else None
            vol_avg_30d = m_metrics.get("VOLUME_AVG_30D") if m_metrics else None
            mom_12_1 = m_metrics.get("MOMENTUM_12_1") if m_metrics else None
            price_vs_52w = m_metrics.get("PRICE_VS_52W_HIGH") if m_metrics else 0.9
            rsi_14 = m_metrics.get("RSI_14_PROXY") if m_metrics else 50

            fwd_ret_1m = m_metrics.get("FWD_RET_1M") if m_metrics else None
            fwd_ret_3m = m_metrics.get("FWD_RET_3M") if m_metrics else None
            fwd_ret_6m = m_metrics.get("FWD_RET_6M") if m_metrics else None
            fwd_ret_12m = m_metrics.get("FWD_RET_12M") if m_metrics else None
            fwd_vol_6m = m_metrics.get("FWD_VOL_6M") if m_metrics else None
            fwd_dd_6m = m_metrics.get("FWD_DD_6M") if m_metrics else None
            triple_barrier = m_metrics.get("TRIPLE_BARRIER") if m_metrics else None

            # Valuation computed from price
            pe = None
            pb = None
            ps = None
            ev_ebitda = None
            ev_sales = None
            earn_yield = None
            fcf_yield = None
            if price_at_fy and eps and eps > 0:
                pe = safe_div(price_at_fy, eps)
                earn_yield = safe_div(1, pe)
            if price_at_fy and bvps and bvps > 0:
                pb = safe_div(price_at_fy, bvps)
            if price_at_fy and rev and shares_d:
                mcap = price_at_fy * shares_d
                ps = safe_div(mcap, rev)
                ev_sales = None
                if net_debt is not None:
                    ev = mcap + net_debt
                    ev_sales = safe_div(ev, rev)
                    if ebitda and ebitda != 0:
                        ev_ebitda = safe_div(ev, ebitda)
            if price_at_fy and fcfps:
                fcf_yield = safe_div(fcfps, price_at_fy)

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
                "PE": pe,
                "PB": pb,
                "PS": ps,
                "EV_EBITDA": ev_ebitda,
                "EV_SALES": ev_sales,
                "EARNINGS_YIELD": earn_yield,
                "FCF_YIELD": fcf_yield,
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
                "RSI_14_PROXY": rsi_14 if rsi_14 is not None else 50,
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
                    "price": price_at_fy,
                }
            )
            fwd_labels.append(
                {
                    "FWD_RET_1M": fwd_ret_1m,
                    "FWD_RET_3M": fwd_ret_3m,
                    "FWD_RET_6M": fwd_ret_6m,
                    "FWD_RET_12M": fwd_ret_12m,
                    "FWD_VOL_6M": fwd_vol_6m,
                    "FWD_DD_6M": fwd_dd_6m,
                    "TRIPLE_BARRIER": triple_barrier,
                    "CEO_CHANGE_FLAG": ceo_change_flag,
                    "TIME_SINCE_CEO_CHANGE": time_since_ceo_change,
                }
            )

    print(
        f"Collected {len(all_rows)} rows from {len({r['ticker'] for r in all_rows})} tickers (v5)"
    )
    D = len(ALL_FEATURES)
    N = len(all_rows)
    Z_raw = np.zeros((N, D), dtype=np.float32)
    mask = np.zeros((N, D), dtype=np.float32)
    tickers = []
    names = []
    fyears = []
    sectors = []
    prices = []
    for i, row in enumerate(all_rows):
        tickers.append(row["ticker"])
        names.append(row["company"])
        fyears.append(row["fiscal_year"])
        sectors.append(row["sector"])
        prices.append(row.get("price") or 0)
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

    # forward labels arrays
    fwd_ret_1m = np.array(
        [
            fl["FWD_RET_1M"] if fl["FWD_RET_1M"] is not None else np.nan
            for fl in fwd_labels
        ],
        dtype=np.float32,
    )
    fwd_ret_3m = np.array(
        [
            fl["FWD_RET_3M"] if fl["FWD_RET_3M"] is not None else np.nan
            for fl in fwd_labels
        ],
        dtype=np.float32,
    )
    fwd_ret_6m = np.array(
        [
            fl["FWD_RET_6M"] if fl["FWD_RET_6M"] is not None else np.nan
            for fl in fwd_labels
        ],
        dtype=np.float32,
    )
    fwd_ret_12m = np.array(
        [
            fl["FWD_RET_12M"] if fl["FWD_RET_12M"] is not None else np.nan
            for fl in fwd_labels
        ],
        dtype=np.float32,
    )
    fwd_vol_6m = np.array(
        [
            fl["FWD_VOL_6M"] if fl["FWD_VOL_6M"] is not None else np.nan
            for fl in fwd_labels
        ],
        dtype=np.float32,
    )
    fwd_dd_6m = np.array(
        [
            fl["FWD_DD_6M"] if fl["FWD_DD_6M"] is not None else np.nan
            for fl in fwd_labels
        ],
        dtype=np.float32,
    )
    triple_barrier = np.array(
        [
            fl["TRIPLE_BARRIER"] if fl["TRIPLE_BARRIER"] is not None else -1
            for fl in fwd_labels
        ],
        dtype=np.int64,
    )
    ceo_change_flag = np.array(
        [fl["CEO_CHANGE_FLAG"] for fl in fwd_labels], dtype=np.float32
    )
    time_since_ceo = np.array(
        [fl["TIME_SINCE_CEO_CHANGE"] for fl in fwd_labels], dtype=np.float32
    )

    # coverage stats
    total_tickers = len(set(tickers))
    total_rows = N
    exec_set_str = {(t, str(fy)) for t, fy in exec_features.keys()}
    rows_with_neo_precise = sum(
        1 for t, fy in zip(tickers, fyears, strict=False) if (t, fy) in exec_set_str
    )
    market_hist_tickers = set(market_hist.keys())
    tickers_with_mhist = len(set(tickers) & market_hist_tickers)
    rows_with_mhist = sum(1 for t in tickers if t in market_hist_tickers)
    form4_keys_str = {(t, str(fy)) for t, fy in form4_flat.keys()}
    rows_with_form4 = sum(
        1 for t, fy in zip(tickers, fyears, strict=False) if (t, fy) in form4_keys_str
    )

    # real flags
    real_flags = {}
    for feat in ALL_FEATURES:
        # determine if real coverage > threshold
        # For market_price and valuation, consider real if market_hist present
        if feat in [
            "RET_1M",
            "RET_3M",
            "RET_6M",
            "RET_12M",
            "VOL_30D",
            "VOL_90D",
            "VOL_252D",
            "BETA_1Y",
            "VOLUME_AVG_30D",
            "MOMENTUM_12_1",
            "PRICE_VS_52W_HIGH",
            "RSI_14_PROXY",
        ]:
            real_flags[feat] = (
                rows_with_mhist / total_rows > 0.5 if total_rows else False
            )
        elif feat in [
            "PE",
            "PB",
            "PS",
            "EV_EBITDA",
            "EV_SALES",
            "EARNINGS_YIELD",
            "FCF_YIELD",
        ]:
            real_flags[feat] = rows_with_mhist / total_rows > 0.3
        elif feat in [
            "NEO_COUNT",
            "CEO_TOTAL_COMP",
            "AVG_NEO_COMP",
            "CEO_TENURE",
            "CEO_FOUNDER_FLAG",
            "NEO_TURNOVER",
            "BOARD_SIZE",
        ]:
            real_flags[feat] = rows_with_neo_precise / total_rows > 0.02
        elif feat == "INSIDER_NET_12M":
            real_flags[feat] = rows_with_form4 / total_rows > 0.05
        else:
            real_flags[feat] = False

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
        "v5": True,
        "years": list(range(2015, 2025)),
        "tickers": len(set(tickers)),
        "rows": N,
        "real_flags": real_flags,
        "real_coverage": {
            "tickers_with_real_neo": len({t for t, _ in exec_features.keys()}),
            "rows_with_real_neo": rows_with_neo_precise,
            "tickers_with_market_hist": tickers_with_mhist,
            "rows_with_market_hist": rows_with_mhist,
            "tickers_with_form4": len({t for t, _ in form4_flat.keys()}),
            "rows_with_form4": rows_with_form4,
        },
        "sources": {
            "def14a_parsed": len(exec_features),
            "form4_index": len(form4_flat),
            "market_history": len(market_hist),
        },
        "forward_labels": [
            "FWD_RET_1M",
            "FWD_RET_3M",
            "FWD_RET_6M",
            "FWD_RET_12M",
            "FWD_VOL_6M",
            "FWD_DD_6M",
            "TRIPLE_BARRIER",
            "CEO_CHANGE_FLAG",
            "TIME_SINCE_CEO_CHANGE",
        ],
    }

    out_v5 = DATA_DIR / "train_matrix_v5.npz"
    np.savez_compressed(
        out_v5,
        Z=Z.astype(np.float32),
        mask=mask,
        ticker=np.array(tickers),
        name=np.array(names),
        fiscal_year=np.array(fyears),
        sector=np.array(sectors),
        cluster=np.zeros(N, dtype=np.int64),
        Z_raw=Z_raw,
        fwd_ret_1m=fwd_ret_1m,
        fwd_ret_3m=fwd_ret_3m,
        fwd_ret_6m=fwd_ret_6m,
        fwd_ret_12m=fwd_ret_12m,
        fwd_vol_6m=fwd_vol_6m,
        fwd_dd_6m=fwd_dd_6m,
        triple_barrier=triple_barrier,
        ceo_change_flag=ceo_change_flag,
        time_since_ceo=time_since_ceo,
        price=np.array(prices, dtype=np.float32),
    )
    # also alias as latest train_matrix
    out_alias = DATA_DIR / "train_matrix.npz"
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

    (DATA_DIR / "feature_manifest_v5.json").write_text(json.dumps(manifest, indent=2))
    (DATA_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    meta = {
        "total_rows": N,
        "total_tickers": total_tickers,
        "avg_rows_per_ticker": N / total_tickers if total_tickers else 0,
        "coverage": manifest["real_coverage"],
        "sources": manifest["sources"],
        "features": ALL_FEATURES,
        "real_flags": real_flags,
        "notes": "v5 adds continuous market history per FY (10y daily), valuation computed from price at FY-end, forward returns 1M-12M, triple barrier, CEO change flag, time since change",
    }
    (DATA_DIR / "real_rows_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved v5 to {out_v5} — N={N} tickers={total_tickers}")
    print(
        f"Forward label coverage: 1M {np.isfinite(fwd_ret_1m).sum()}/{N} 3M {np.isfinite(fwd_ret_3m).sum()} 6M {np.isfinite(fwd_ret_6m).sum()} barrier {np.sum(triple_barrier != -1)}"
    )
    print(
        f"Market hist coverage {rows_with_mhist}/{N} {rows_with_mhist / N * 100:.1f}%"
    )
    print(f"Real flags true: {[k for k, v in real_flags.items() if v]}")


if __name__ == "__main__":
    build_v5()
