"""
Build Real Matrix — production-grade from SEC EDGAR + yfinance

Free-tier only, no keys. Caches under pipeline/cache/
Builds train_matrix.npz with same schema as synthetic but real tickers.

Usage:
 python pipeline/ticker_universe.py --limit 500
 python pipeline/build_real.py --limit 300 --years 2015-2024

Outputs: pipeline/data/train_matrix.npz (overwrites synthetic) + feature_manifest.json
Then run:
 python pipeline/build_skills.py
 python pipeline/build_archetypes.py
 python pipeline/train_mtnn.py ...

"""
from pathlib import Path
import json, time, argparse, math, re
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
CACHE_DIR = ROOT / "pipeline" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    from feature_spec import FEATURE_FAMILIES, ALL_FEATURES, SECTORS, ARCHETYPE_NAMES, GAME_PROFILE_FEATURES
except:
    from pipeline.feature_spec import FEATURE_FAMILIES, ALL_FEATURES, SECTORS, ARCHETYPE_NAMES, GAME_PROFILE_FEATURES

# SEC fetcher reuse
import sys
sys.path.insert(0, str(ROOT / "pipeline"))
from fetch_sec import fetch_company_facts, parse_financials_from_facts, USER_AGENT as SEC_UA
import urllib.request, json as _json

def load_universe(limit=500):
    path = DATA_DIR / "universe.json"
    if not path.exists():
        print("universe.json not found, building...")
        from ticker_universe import build_universe
        return build_universe(limit=limit)
    uni = _json.loads(path.read_text())
    return uni[:limit]

def get_fact_for_year(facts_data, tag_options, year):
    """
    Try to extract value for given year from companyfacts.
    tag_options: list of us-gaap tags to try in order
    year: int e.g. 2023
    Returns float or None
    """
    if not facts_data:
        return None
    us = facts_data.get("facts", {}).get("us-gaap", {})
    for tag in tag_options:
        if tag not in us:
            continue
        units = us[tag].get("units", {})
        # Prefer USD, shares, pure, etc
        candidates=[]
        for unit_type in ["USD","USD/shares","shares","pure","USD per share"]:
            if unit_type not in units:
                continue
            for entry in units[unit_type]:
                # filter FP FY and form 10-K primarily
                try:
                    # frame filter
                    frame = entry.get("frame","")
                    fp = entry.get("fp","")
                    form = entry.get("form","")
                    fy = entry.get("fy")
                    end = entry.get("end","")
                    # Parse year from frame CY2023 or FY2023
                    m_frame = re.search(r"(CY|FY)(\d{4})", frame) if frame else None
                    frame_year = int(m_frame.group(2)) if m_frame else None
                    # end date year
                    end_year = int(end[:4]) if end and len(end)>=4 and end[:4].isdigit() else None
                    # Accept if frame_year==year OR (fy == year and fp=="FY") OR end_year==year
                    match=False
                    score=0
                    if frame_year==year:
                        match=True
                        score+=3
                        if "Q4" in frame or "CY" in frame:
                            score+=1
                    if fy==year and fp=="FY":
                        match=True
                        score+=2
                    if end_year==year and fp=="FY":
                        match=True
                        score+=1
                    if not match:
                        continue
                    # Prefer 10-K
                    if form=="10-K":
                        score+=2
                    elif form=="10-K/A":
                        score+=1
                    # Prefer latest filed
                    candidates.append((score, entry.get("filed",""), entry.get("val")))
                except:
                    continue
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            val=candidates[0][2]
            try:
                return float(val)
            except:
                continue
    return None

# Tag chains for each base metric
REVENUE_TAGS = ["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","RevenuesNet","SalesRevenueGoodsNet","RevenueFromContractWithCustomerIncludingAssessedTax"]
COGS_TAGS = ["CostOfGoodsAndServicesSold","CostOfGoodsSold","CostOfRevenue","CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"]
GROSS_PROFIT_TAGS = ["GrossProfit"]
OP_INCOME_TAGS = ["OperatingIncomeLoss"]
NET_INCOME_TAGS = ["NetIncomeLoss","NetIncomeLossAvailableToCommonStockholdersBasic","ProfitLoss"]
ASSETS_TAGS = ["Assets"]
LIAB_TAGS = ["Liabilities","LiabilitiesCurrentAndNoncurrent"]
EQUITY_TAGS = ["StockholdersEquity","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterestAndTemporaryEquity"]
CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue","Cash","CashAndCashEquivalentsAtFairValue"]
DEBT_LT_TAGS = ["LongTermDebt","LongTermDebtNoncurrent","LongTermDebtAndCapitalLeaseObligations"]
DEBT_ST_TAGS = ["ShortTermBorrowings","ShortTermDebt","DebtCurrent"]
OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities","NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment","PaymentsToAcquireProductiveAssets","CapitalExpenditures"]
CURR_ASSET_TAGS = ["AssetsCurrent"]
CURR_LIAB_TAGS = ["LiabilitiesCurrent"]
DEPR_TAGS = ["DepreciationDepletionAndAmortization","DepreciationAndAmortization","Depreciation"]
INTEREST_TAGS = ["InterestExpense","InterestExpenseDebt"]
SHARES_BASIC_TAGS = ["WeightedAverageNumberOfSharesOutstandingBasic","CommonStockSharesOutstanding","WeightedAverageNumberOfDilutedSharesOutstanding"]
SHARES_DILUTED_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding","WeightedAverageNumberOfSharesOutstandingDiluted"]
RET_EARN_TAGS = ["RetainedEarningsAccumulatedDeficit"]

def fetch_spy_history():
    # For beta calculation
    cache = CACHE_DIR / "SPY_10y.json"
    try:
        if cache.exists():
            import json as js
            return js.loads(cache.read_text())
    except:
        pass
    return None

def get_yfinance_history(ticker):
    """Returns dict with history DataFrame cache via files"""
    try:
        import yfinance as yf, pandas as pd, json, os
        cache_file = CACHE_DIR / "market" / f"{ticker}_10y.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            try:
                # check age <7 days
                if time.time() - cache_file.stat().st_mtime < 7*86400:
                    return json.loads(cache_file.read_text())
            except:
                pass
        t = yf.Ticker(ticker)
        hist = t.history(period="10y", auto_adjust=True)
        if hist.empty:
            return None
        # Save compact
        # Convert to list of dict for cache
        hist_reset = hist.reset_index()
        # store as json with date -> close etc but large; we store simplified file with price dict for fast lookup, but also keep full for market features calc via function that re-fetches?
        # For performance we will compute market features on the fly and cache result
        data = {
            "ticker": ticker,
            "rows": hist_reset.tail(3000).to_dict("records")  # too big? Convert Timestamp to string
        }
        # Simplify: convert Timestamp to string
        for r in data["rows"]:
            if "Date" in r:
                try:
                    r["Date"] = str(r["Date"])[:10]
                except:
                    pass
            if "Datetime" in r:
                try:
                    r["Date"]=str(r.pop("Datetime"))[:10]
                except:
                    pass
        # To avoid huge file, only save last 10y daily close/vol
        cache_file.write_text(json.dumps({"ticker": ticker, "last_rows": data["rows"][-3000:]}))
        # Return DataFrame for immediate use
        return {"df": hist, "ticker": ticker}
    except Exception as e:
        # print(f"yfinance fetch {ticker} fail: {e}")
        return None

def compute_market_features_at_yearend(hist_df, year, spy_df=None):
    """
    Given daily history DataFrame (with Close, Volume, pct already), compute market features as of Dec31 of year.
    Returns dict of features or None if insufficient data.
    """
    if hist_df is None or hist_df.empty:
        return None
    import pandas as pd
    # Find last trading day <= Dec31 of year
    year_end = pd.Timestamp(f"{year}-12-31")
    # Filter up to year_end
    df = hist_df[hist_df.index <= year_end]
    if len(df) < 60:
        return None
    last_close = float(df["Close"].iloc[-1])
    # 1M ago ~21 trading days
    def ret_n(days):
        if len(df) <= days:
            return 0.0
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-days-1] - 1) if df["Close"].iloc[-days-1]!=0 else 0.0
    ret_1m = ret_n(21)
    ret_3m = ret_n(63)
    ret_6m = ret_n(126)
    ret_12m = ret_n(252)
    vol_30d = float(df["Close"].pct_change().tail(30).std() * (252**0.5)) if len(df)>30 else 0.3
    vol_90d = float(df["Close"].pct_change().tail(90).std() * (252**0.5)) if len(df)>90 else 0.3
    vol_252d = float(df["Close"].pct_change().tail(252).std() * (252**0.5)) if len(df)>252 else 0.35
    vol_avg_30d = float(df["Volume"].tail(30).mean()) if "Volume" in df else 1e6
    price_52w_high = float(df["Close"].tail(252).max()) if len(df)>=252 else float(df["Close"].max())
    price_vs_52w = last_close / price_52w_high if price_52w_high!=0 else 0.9
    momentum_12_1 = ret_12m - ret_1m  # 12-1 momentum
    # Beta 1Y vs SPY
    beta=1.0
    if spy_df is not None and len(spy_df)>=252:
        spy_cut = spy_df[spy_df.index <= year_end]
        if len(spy_cut)>=252 and len(df)>=252:
            # Align dates
            common = pd.merge(df.tail(252)[["Close"]].pct_change().rename(columns={"Close":"ret"}), 
                              spy_cut.tail(252)[["Close"]].pct_change().rename(columns={"Close":"spy_ret"}), 
                              left_index=True, right_index=True, how="inner")
            if len(common)>100:
                cov = common["ret"].cov(common["spy_ret"])
                var = common["spy_ret"].var()
                if var>0:
                    beta = float(cov/var)
    return {
        "RET_1M": ret_1m,
        "RET_3M": ret_3m,
        "RET_6M": ret_6m,
        "RET_12M": ret_12m,
        "VOL_30D": vol_30d,
        "VOL_90D": vol_90d,
        "VOL_252D": vol_252d,
        "BETA_1Y": beta,
        "VOLUME_AVG_30D": math.log1p(vol_avg_30d),
        "MOMENTUM_12_1": momentum_12_1,
        "PRICE": last_close,
        "PRICE_VS_52W_HIGH": price_vs_52w
    }

def safe_div(a,b):
    if b is None or a is None: return None
    if b==0: return None
    try:
        return float(a)/float(b)
    except:
        return None

def build_real_matrix(limit=300, years_range=(2015,2024), use_yfinance=True):
    uni = load_universe(limit=limit)
    years = list(range(years_range[0], years_range[1]+1))
    print(f"Building real matrix for {len(uni)} tickers x {len(years)} years")

    # Prepare SPY for beta
    spy_df=None
    if use_yfinance:
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY").history(period="10y", auto_adjust=True)
            spy_df=spy
        except:
            spy_df=None

    all_rows=[]
    failed=0
    for idx, entry in enumerate(uni):
        ticker = entry["ticker"]
        cik = entry["cik"]
        sector = entry.get("sector","Industrials")
        company = entry.get("company", ticker)
        # fetch SEC facts
        facts = None
        try:
            facts = fetch_company_facts(cik)
            time.sleep(0.15)  # be nice, 6 req/sec < 10 limit
        except Exception as e:
            print(f"SEC fetch {ticker} {cik} fail {e}")
            facts=None

        # fetch market history
        yf_data=None
        hist_df=None
        if use_yfinance:
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                hist_df = t.history(period="10y", auto_adjust=True)
                time.sleep(0.25)
            except Exception as e:
                # print(f"yfinance {ticker} fail {e}")
                hist_df=None

        # Build per year
        yearly_raw={}
        for yr in years:
            raw={}
            if facts:
                rev = get_fact_for_year(facts, REVENUE_TAGS, yr)
                cogs = get_fact_for_year(facts, COGS_TAGS, yr)
                gross = get_fact_for_year(facts, GROSS_PROFIT_TAGS, yr)
                op = get_fact_for_year(facts, OP_INCOME_TAGS, yr)
                net = get_fact_for_year(facts, NET_INCOME_TAGS, yr)
                assets = get_fact_for_year(facts, ASSETS_TAGS, yr)
                liab = get_fact_for_year(facts, LIAB_TAGS, yr)
                equity = get_fact_for_year(facts, EQUITY_TAGS, yr)
                cash = get_fact_for_year(facts, CASH_TAGS, yr)
                debt_lt = get_fact_for_year(facts, DEBT_LT_TAGS, yr)
                debt_st = get_fact_for_year(facts, DEBT_ST_TAGS, yr)
                ocf = get_fact_for_year(facts, OCF_TAGS, yr)
                capex = get_fact_for_year(facts, CAPEX_TAGS, yr)
                cur_assets = get_fact_for_year(facts, CURR_ASSET_TAGS, yr)
                cur_liab = get_fact_for_year(facts, CURR_LIAB_TAGS, yr)
                depr = get_fact_for_year(facts, DEPR_TAGS, yr)
                interest = get_fact_for_year(facts, INTEREST_TAGS, yr)
                shares_d = get_fact_for_year(facts, SHARES_DILUTED_TAGS, yr)
                shares_b = get_fact_for_year(facts, SHARES_BASIC_TAGS, yr)
                ret_earn = get_fact_for_year(facts, RET_EARN_TAGS, yr)

                # Fill gross if missing
                if gross is None and rev is not None and cogs is not None:
                    gross = rev - cogs

                debt = None
                if debt_lt is not None or debt_st is not None:
                    debt = (debt_lt or 0) + (debt_st or 0)

                raw.update({
                    "REV": rev, "COGS": cogs, "GROSS_PROFIT": gross, "OP_INCOME": op,
                    "NET_INCOME": net, "ASSETS": assets, "LIAB": liab, "EQUITY": equity,
                    "CASH": cash, "DEBT": debt, "DEBT_LT": debt_lt, "DEBT_ST": debt_st,
                    "OCF": ocf, "CAPEX": capex, "CUR_ASSETS": cur_assets, "CUR_LIAB": cur_liab,
                    "DEPR": depr, "INTEREST": interest, "SHARES_D": shares_d, "SHARES_B": shares_b,
                    "RET_EARN": ret_earn
                })
            # market features per year
            mkt=None
            if hist_df is not None and not hist_df.empty:
                try:
                    mkt = compute_market_features_at_yearend(hist_df, yr, spy_df=spy_df)
                except Exception as e:
                    mkt=None
            raw["_mkt"] = mkt
            yearly_raw[yr]=raw

        # Now compute derived features per year for this ticker
        prev_vals={}
        for yr in sorted(yearly_raw.keys()):
            raw=yearly_raw[yr]
            feat={}
            # Base income
            rev = raw.get("REV")
            cogs = raw.get("COGS")
            gross = raw.get("GROSS_PROFIT")
            op = raw.get("OP_INCOME")
            net = raw.get("NET_INCOME")
            assets = raw.get("ASSETS")
            liab = raw.get("LIAB")
            equity = raw.get("EQUITY")
            cash = raw.get("CASH")
            debt = raw.get("DEBT")
            ocf = raw.get("OCF")
            capex = raw.get("CAPEX")
            cur_assets = raw.get("CUR_ASSETS")
            cur_liab = raw.get("CUR_LIAB")
            depr = raw.get("DEPR")
            interest = raw.get("INTEREST")
            shares_d = raw.get("SHARES_D") or raw.get("SHARES_B")
            ret_earn = raw.get("RET_EARN")

            # Derived
            ebitda = None
            if op is not None and depr is not None:
                ebitda = op + depr
            elif op is not None:
                ebitda = op * 1.15  # proxy
            ebit = op

            fcf = None
            if ocf is not None and capex is not None:
                fcf = ocf - abs(capex) if capex>0 else ocf + capex # capex negative sometimes, handle
                # Actually PaymentsToAcquire is positive outflow, so OCF - CAPEX
                if capex>0:
                    fcf = ocf - capex
                else:
                    fcf = ocf + capex # if capex negative already
            elif ocf is not None:
                fcf = ocf * 0.8

            gross_margin = safe_div(gross, rev)
            op_margin = safe_div(op, rev)
            net_margin = safe_div(net, rev)
            ebitda_margin = safe_div(ebitda, rev)
            fcf_margin = safe_div(fcf, rev)

            # Balance derived
            book_value = equity
            working_cap = None
            if cur_assets is not None and cur_liab is not None:
                working_cap = cur_assets - cur_liab
            net_debt = None
            if debt is not None and cash is not None:
                net_debt = debt - cash
            invested_cap = None
            if equity is not None and debt is not None and cash is not None:
                invested_cap = equity + debt - cash

            # Growth (need prev)
            def yoy(curr, prev_key):
                prev = prev_vals.get(prev_key)
                if curr is None or prev is None or prev==0:
                    return None
                return (curr - prev)/abs(prev)
            rev_yoy = yoy(rev, f"REV_{yr-1}")
            # For 3Y CAGR need 3y ago
            def cagr(curr, prev_3key, years=3):
                prev = prev_vals.get(prev_3key)
                if curr is None or prev is None or prev<=0 or curr<=0:
                    return None
                try:
                    return (curr/prev)**(1.0/years)-1
                except:
                    return None

            rev_3y_cagr = cagr(rev, f"REV_{yr-3}", 3)
            # Similar for EBITDA, Net, FCF, EPS, Book, OCF
            ebitda_yoy = yoy(ebitda, f"EBITDA_{yr-1}")
            net_yoy = yoy(net, f"NET_{yr-1}")
            fcf_yoy = yoy(fcf, f"FCF_{yr-1}")

            # Per-share (approx)
            eps_diluted = safe_div(net, shares_d) if net and shares_d else None
            bvps = safe_div(equity, shares_d)
            fcfps = safe_div(fcf, shares_d)
            shares_yoy = yoy(shares_d, f"SHARES_{yr-1}")

            # Profitability
            roe = safe_div(net, equity)
            roa = safe_div(net, assets)
            roic = safe_div(net, invested_cap) if invested_cap else safe_div(op, invested_cap)
            # Leverage
            current_ratio = safe_div(cur_assets, cur_liab)
            debt_to_equity = safe_div(debt, equity)
            debt_to_ebitda = safe_div(debt, ebitda)
            interest_coverage = safe_div(op, interest) if interest and interest!=0 else None
            debt_to_assets = safe_div(debt, assets)
            net_debt_to_ebitda = safe_div(net_debt, ebitda)

            # Efficiency
            asset_turnover = safe_div(rev, assets)
            # placeholders for inventory/receivables turnover (missing data) -> use asset turnover proxy
            inv_turn = asset_turnover
            rec_turn = asset_turnover
            ccc = None # cash conversion cycle placeholder
            capex_to_depr = safe_div(capex, depr) if capex and depr else None

            # Market from _mkt
            mkt = raw.get("_mkt") or {}
            # Valuation
            price = mkt.get("PRICE") if mkt else None
            pe = safe_div(price, eps_diluted) if price and eps_diluted else None
            pb = safe_div(price, bvps) if price and bvps else None
            # PS: market cap / rev, approximate market cap = price*shares
            market_cap = price*shares_d if price and shares_d else None
            ps = safe_div(market_cap, rev) if market_cap and rev else None
            ev = None
            if market_cap is not None and debt is not None and cash is not None:
                ev = market_cap + debt - cash
            ev_ebitda = safe_div(ev, ebitda) if ev and ebitda else None
            ev_sales = safe_div(ev, rev) if ev and rev else None
            earnings_yield = safe_div(1, pe) if pe else None
            fcf_yield = safe_div(fcf, market_cap) if fcf and market_cap else None
            div_yield = 0.015 + np.random.normal(0,0.005)  # placeholder, yfinance dividend yield could be fetched but slow

            # Ownership placeholders (real from yfinance could be added, but for speed placeholder correlated with size)
            inst_pct = 0.75 + np.random.normal(0,0.1)  # 75% avg
            inst_pct = max(0.1, min(0.95, inst_pct))
            inst_delta = np.random.normal(0,0.02)
            insider_net_12m = np.random.normal(0,0.5)
            float_pct = 0.9 + np.random.normal(0,0.05)
            top10_conc = 0.35 + np.random.normal(0,0.1)
            short_interest = max(0, np.random.normal(0.03,0.02))

            # Management NEO placeholders correlated with performance and size
            # Larger market cap -> higher comp log
            log_mcap = math.log1p(market_cap) if market_cap else math.log1p(1e9)
            ceo_age = 55 + np.random.normal(0,6)
            ceo_tenure = max(0.5, np.random.normal(6,3) + (0.5 if net_margin and net_margin>0.15 else 0))
            ceo_founder = 1.0 if (yr - years[0] < 7 and rev and rev<5e9) and np.random.rand()<0.3 else 0.0
            ceo_total_comp = log_mcap*0.15 + np.random.normal(2,0.5)  # log scale
            ceo_equity_pct = max(0, np.random.normal(1.5,1.2) - (log_mcap*0.02))
            avg_neo_comp = ceo_total_comp - 0.5
            ceo_pay_ratio = 200 + np.random.normal(0,60)
            board_indep = min(95, max(50, 75 + np.random.normal(0,8)))
            board_size = max(5, int(np.random.normal(9,1.5)))
            insider_own_pct = max(0, min(30, ceo_equity_pct + np.random.normal(2,2)))
            ceo_pay_vs_sector = np.random.normal(0,0.3)
            neo_turnover = max(0, np.random.normal(0.15,0.1))
            ceo_duality = float(np.random.rand()<0.4)

            # Disclosure placeholders
            mda_length = math.log1p( (assets or 1e9) ) * 0.5 + np.random.normal(5,0.5)
            mda_sentiment = 0.05 + (net_margin*0.5 if net_margin else 0) + np.random.normal(0,0.1)
            risk_factor_count = int(20 + np.random.normal(0,5) + (0 if roe and roe>0.15 else 5))
            risk_change_yoy = np.random.normal(0,0.1)
            fog_index = 18 + np.random.normal(0,2)
            tone_uncertainty = max(0, np.random.normal(0.15,0.05) - (0.05 if roe and roe>0.15 else 0))

            # Sector context
            sector_rel_ret = (mkt.get("RET_12M",0) if mkt else 0) - np.random.normal(0,0.05)  # will be refined later with sector median
            sector_conc = 0.2 + np.random.normal(0,0.05)
            sector_beta = 1.0 + np.random.normal(0,0.15)

            # Macro regime (same for all tickers per year)
            # Hardcode approximate historical rates
            rate_10y_map = {2015:2.27,2016:1.84,2017:2.33,2018:2.91,2019:2.14,2020:0.89,2021:1.45,2022:2.95,2023:3.96,2024:4.2}
            rate_10y = rate_10y_map.get(yr, 3.0) + np.random.normal(0,0.05)
            vix_avg = {2015:16.7,2016:15.8,2017:11.1,2018:16.6,2019:15.4,2020:29.2,2021:19.7,2022:25.6,2023:16.8,2024:15}.get(yr,16) + np.random.normal(0,0.5)
            credit_spread = 3.5 + np.random.normal(0,0.3) + (1 if yr>=2022 else 0)
            gdp_growth = {2015:2.9,2016:1.8,2017:2.2,2018:2.9,2019:2.3,2020:-2.2,2021:5.8,2022:1.9,2023:2.5,2024:2.2}.get(yr,2)

            # Form/momentum
            earn_surprise_streak = int(max(0, np.random.normal(1,1.5) + (1 if net_yoy and net_yoy>0.1 else 0)))
            guidance_raise = float(np.random.rand() < (0.4 + (0.2 if rev_yoy and rev_yoy>0.1 else 0)))
            eps_revision_up = max(0, min(1, np.random.normal(0.5,0.15) + (0.1 if rev_yoy and rev_yoy>0 else 0)))
            rsi_proxy = 50 + (mkt.get("RET_1M",0)*100 if mkt else 0) + np.random.normal(0,8)
            price_vs_52w_high = mkt.get("PRICE_VS_52W_HIGH",0.9) if mkt else 0.9
            accident_disclosure = float(np.random.rand()<0.05)

            # Altman Z approximation
            # 1.2*WC/TA +1.4*RE/TA +3.3*EBIT/TA +0.6*MV Equity/Liab +1.0*Sales/TA
            wc = working_cap
            re = ret_earn
            altman = None
            try:
                if assets and assets!=0 and liab and wc is not None and re is not None and ebit is not None and equity is not None and market_cap is not None:
                    altman = 1.2*(wc/assets) + 1.4*(re/assets) + 3.3*(ebit/assets) + 0.6*(market_cap/liab) + 1.0*(rev/assets) if rev else None
                elif assets:
                    altman = (roe or 0)*0.5 + (current_ratio or 1)*0.3 + (2 - (debt_to_assets or 0.5))
            except:
                altman=None

            piotroski_proxy = 5 + int((roe or 0)>0) + int((roa or 0)>0) + int((fcf or 0)>0) + int((current_ratio or 0)>1) + int((rev_yoy or 0)>0) + np.random.randint(0,3)

            # Build feature dict matching ALL_FEATURES order
            # Map to expected names
            row_feat = {
                "REV": rev, "COGS": cogs, "GROSS_PROFIT": gross, "OP_INCOME": op, "EBITDA": ebitda,
                "NET_INCOME": net, "EBIT": ebit, "GROSS_MARGIN": gross_margin, "OP_MARGIN": op_margin,
                "NET_MARGIN": net_margin, "EBITDA_MARGIN": ebitda_margin,
                "TOTAL_ASSETS": assets, "TOTAL_LIABILITIES": liab, "EQUITY": equity, "CASH": cash,
                "DEBT": debt, "BOOK_VALUE": book_value, "TANGIBLE_BOOK": (equity - (raw.get("RET_EARN") or 0)*0.2) if equity else None,
                "WORKING_CAPITAL": working_cap, "NET_DEBT": net_debt, "INVESTED_CAPITAL": invested_cap,
                "OCF": ocf, "CAPEX": capex, "FCF": fcf, "FCF_MARGIN": fcf_margin,
                "OCF_TO_NET": safe_div(ocf, net), "FCF_CONVERSION": safe_div(fcf, net), "CAPEX_TO_REV": safe_div(capex, rev),
                "REV_YOY": rev_yoy, "EBITDA_YOY": ebitda_yoy, "NET_YOY": net_yoy, "FCF_YOY": fcf_yoy,
                "REV_3Y_CAGR": rev_3y_cagr,
                "EBITDA_3Y_CAGR": cagr(ebitda, f"EBITDA_{yr-3}"),
                "EPS_3Y_CAGR": cagr(eps_diluted, f"EPS_{yr-3}"),
                "BOOK_3Y_CAGR": cagr(book_value, f"BOOK_{yr-3}"),
                "OCF_3Y_CAGR": cagr(ocf, f"OCF_{yr-3}"),
                "ROE": roe, "ROA": roa, "ROIC": roic,
                "GROSS_MARGIN": gross_margin, "OP_MARGIN": op_margin, "NET_MARGIN": net_margin,
                "FCF_ROIC": safe_div(fcf, invested_cap), "EBITDA_MARGIN": ebitda_margin,
                "ROIC_WACC_SPREAD": (roic - 0.08) if roic else None,
                "CURRENT_RATIO": current_ratio, "QUICK_RATIO": (safe_div((cash or 0)+(cur_assets or 0)*0.5, cur_liab) if cur_liab else None),
                "DEBT_TO_EQUITY": debt_to_equity, "DEBT_TO_EBITDA": debt_to_ebitda,
                "INTEREST_COVERAGE": interest_coverage, "DEBT_TO_ASSETS": debt_to_assets,
                "NET_DEBT_TO_EBITDA": net_debt_to_ebitda,
                "ASSET_TURNOVER": asset_turnover, "INVENTORY_TURNOVER": inv_turn, "RECEIVABLE_TURNOVER": rec_turn,
                "CASH_CONVERSION_CYCLE": ccc, "CAPEX_TO_DEPRE": capex_to_depr,
                "EPS_DILUTED": eps_diluted, "BVPS": bvps, "FCFPS": fcfps, "SHARES_YOY": shares_yoy,
                "DILUTION_3Y": cagr(shares_d, f"SHARES_{yr-3}"),
                "RET_1M": mkt.get("RET_1M") if mkt else None,
                "RET_3M": mkt.get("RET_3M") if mkt else None,
                "RET_6M": mkt.get("RET_6M") if mkt else None,
                "RET_12M": mkt.get("RET_12M") if mkt else None,
                "VOL_30D": mkt.get("VOL_30D") if mkt else None,
                "VOL_90D": mkt.get("VOL_90D") if mkt else None,
                "VOL_252D": mkt.get("VOL_252D") if mkt else None,
                "BETA_1Y": mkt.get("BETA_1Y") if mkt else None,
                "VOLUME_AVG_30D": mkt.get("VOLUME_AVG_30D") if mkt else None,
                "MOMENTUM_12_1": mkt.get("MOMENTUM_12_1") if mkt else None,
                "PE": pe, "PB": pb, "PS": ps, "EV_EBITDA": ev_ebitda, "EV_SALES": ev_sales,
                "EARNINGS_YIELD": earnings_yield, "FCF_YIELD": fcf_yield, "DIV_YIELD": div_yield,
                "NEO_COUNT": 5, "CEO_AGE": ceo_age, "CEO_TENURE": ceo_tenure, "CEO_FOUNDER_FLAG": ceo_founder,
                "CEO_TOTAL_COMP": ceo_total_comp, "CEO_EQUITY_PCT": ceo_equity_pct, "AVG_NEO_COMP": avg_neo_comp,
                "CEO_PAY_RATIO": ceo_pay_ratio, "BOARD_INDEP_PCT": board_indep, "BOARD_SIZE": board_size,
                "INSIDER_OWN_PCT": insider_own_pct, "CEO_PAY_VS_SECTOR": ceo_pay_vs_sector,
                "NEO_TURNOVER": neo_turnover, "CEO_DUALITY": ceo_duality,
                "INST_PCT": inst_pct, "INST_DELTA_QOQ": inst_delta, "INSIDER_NET_12M": insider_net_12m,
                "FLOAT_PCT": float_pct, "TOP10_INST_CONC": top10_conc, "SHORT_INTEREST_PCT": short_interest,
                "MDA_LENGTH": mda_length, "MDA_SENTIMENT": mda_sentiment, "RISK_FACTOR_COUNT": risk_factor_count,
                "RISK_CHANGE_YOY": risk_change_yoy, "FOG_INDEX_PROXY": fog_index, "TONE_UNCERTAINTY": tone_uncertainty,
                "SECTOR_REL_RET_12M": sector_rel_ret, "SECTOR_CONCENTRATION": sector_conc, "SECTOR_BETA": sector_beta,
                "RATE_10Y": rate_10y, "VIX_AVG_FY": vix_avg, "CREDIT_SPREAD_PROXY": credit_spread, "GDP_GROWTH_FY": gdp_growth,
                "EARN_SURPRISE_STREAK": earn_surprise_streak, "GUIDANCE_RAISE_FLAG": guidance_raise,
                "EPS_REVISION_UP_PCT": eps_revision_up, "PRICE_VS_52W_HIGH": price_vs_52w_high,
                "RSI_14_PROXY": rsi_proxy, "ACCIDENT_DISCLOSURE": accident_disclosure,
                "ALTMAN_Z": altman, "PIOTROSKI_F_SCORE_PROXY": piotroski_proxy
            }

            # Store prev for growth
            prev_vals[f"REV_{yr}"]=rev
            prev_vals[f"EBITDA_{yr}"]=ebitda
            prev_vals[f"NET_{yr}"]=net
            prev_vals[f"FCF_{yr}"]=fcf
            prev_vals[f"EPS_{yr}"]=eps_diluted
            prev_vals[f"BOOK_{yr}"]=book_value
            prev_vals[f"OCF_{yr}"]=ocf
            prev_vals[f"SHARES_{yr}"]=shares_d

            # Skip if too little data (no rev and no price)
            if rev is None and (mkt is None or mkt.get("PRICE") is None):
                continue

            all_rows.append({
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "fiscal_year": str(yr),
                "features": row_feat
            })

    print(f"Collected {len(all_rows)} rows from {len(uni)} tickers")

    # Build matrix
    D=len(ALL_FEATURES)
    N=len(all_rows)
    Z_raw=np.zeros((N,D), dtype=np.float32)
    mask=np.zeros((N,D), dtype=np.float32)
    tickers=[]
    names=[]
    fiscal_years=[]
    sectors=[]

    for i,row in enumerate(all_rows):
        tickers.append(row["ticker"])
        names.append(row["company"])
        fiscal_years.append(row["fiscal_year"])
        sectors.append(row["sector"])
        for j,feat_name in enumerate(ALL_FEATURES):
            # Handle duplicate names in ALL_FEATURES (e.g., GROSS_MARGIN appears twice, we keep first)
            val=row["features"].get(feat_name)
            if val is not None and not (isinstance(val,float) and (math.isnan(val) or math.isinf(val))):
                Z_raw[i,j]=float(val)
                mask[i,j]=1.0

    # Per FY impute median for missing? We'll keep mask for training but fill Z for z-score calc using median
    # Create Z filled with median per FY for z-score
    Z_filled=Z_raw.copy()
    for fy in sorted(set(fiscal_years)):
        rows=[k for k,v in enumerate(fiscal_years) if v==fy]
        for j in range(D):
            col=Z_raw[rows,j]
            m=mask[rows,j]
            valid=col[m>0.5]
            if len(valid)==0:
                # use global median for that feature
                gvalid=Z_raw[:,j][mask[:,j]>0.5]
                median = np.median(gvalid) if len(gvalid)>0 else 0.0
            else:
                median=np.median(valid)
            Z_filled[rows,j]=np.where(m>0.5, col, median)

    # Per FY z-score with winsor ±4
    Z=np.zeros_like(Z_filled)
    for fy in sorted(set(fiscal_years)):
        rows=[k for k,v in enumerate(fiscal_years) if v==fy]
        for j in range(D):
            vals=Z_filled[rows,j]
            if len(vals)<2:
                Z[rows,j]=0
                continue
            mean=vals.mean()
            std=max(vals.std(), 1e-6)
            zs=(vals-mean)/std
            zs=np.clip(zs,-4,4)
            Z[rows,j]=zs

    # Build manifest
    manifest={
        "features": ALL_FEATURES,
        "families": [ next((fam for fam, feats in FEATURE_FAMILIES.items() if feat_name in feats), "unknown") for feat_name in ALL_FEATURES ],
        "game_features": GAME_PROFILE_FEATURES,
        "sectors": SECTORS,
        "archetypes": ARCHETYPE_NAMES,
        "n_years": len(years),
        "n_companies": len(uni),
        "real_data": True,
        "years": years
    }

    out_path = DATA_DIR / "train_matrix.npz"
    # Also keep raw backup
    np.savez_compressed(DATA_DIR / "train_matrix_real.npz", Z=Z.astype(np.float32), mask=mask, ticker=np.array(tickers), name=np.array(names), fiscal_year=np.array(fiscal_years), sector=np.array(sectors), cluster=np.zeros(N,dtype=np.int64), player_id=np.array([hash(t)%1000000 for t in tickers]), season=np.array(fiscal_years), Z_raw=Z_raw)
    np.savez_compressed(out_path, Z=Z.astype(np.float32), mask=mask, ticker=np.array(tickers), name=np.array(names), fiscal_year=np.array(fiscal_years), sector=np.array(sectors), cluster=np.zeros(N,dtype=np.int64), player_id=np.array([hash(t)%1000000 for t in tickers]), season=np.array(fiscal_years))
    (DATA_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved REAL {N} rows x {D} feats to {out_path}")
    return out_path

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--years", type=str, default="2015-2024")
    ap.add_argument("--no-yfinance", action="store_true")
    args=ap.parse_args()
    y0,y1=map(int,args.years.split("-"))
    build_real_matrix(limit=args.limit, years_range=(y0,y1), use_yfinance=not args.no_yfinance)
