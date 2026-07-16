"""
SEC EDGAR fetcher — free-tier only, public domain filings
Fetches 10-K, 10-Q, DEF 14A cached JSON

Uses https://data.sec.gov/api/xbrl/companyfacts + submissions
No key required, respects rate limit (10 req/sec with User-Agent)

This is scaffold for real pipeline; offline synthetic still works.
Solo personal project, no connection to employer, built with public/free-tier only
"""
from pathlib import Path
import json, time, os
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "sec"
CACHE.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Cameron Davis jcdavis131@gmail.com VectorEquities research"

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            time.sleep(1+attempt*2)
            if attempt==2:
                print(f"SEC fetch fail {url}: {e}")
                return None

def fetch_company_facts(cik: str):
    # cik zero-padded 10 digits
    cik_pad = str(cik).zfill(10)
    cache_file = CACHE / f"facts_{cik_pad}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_pad}.json"
    data = fetch_json(url)
    if data:
        cache_file.write_text(json.dumps(data)[:5000000])  # truncate for cache size
    time.sleep(0.2)
    return data

def parse_financials_from_facts(facts):
    """Extract key us-gaap metrics: Revenues, NetIncomeLoss, Assets etc"""
    if not facts:
        return {}
    # Simplified extraction
    out={}
    try:
        us = facts.get("facts", {}).get("us-gaap", {})
        for key in ["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet",
                    "NetIncomeLoss","GrossProfit","OperatingIncomeLoss",
                    "Assets","Liabilities","StockholdersEquity","CashAndCashEquivalentsAtCarryingValue",
                    "LongTermDebt","EarningsPerShareDiluted"]:
            if key in us:
                units = us[key].get("units", {})
                # prefer USD or shares
                for unit in ["USD","USD/shares","shares"]:
                    if unit in units and units[unit]:
                        # latest
                        latest = sorted(units[unit], key=lambda x: x.get("end",""))[-1]
                        out[key]=latest.get("val")
                        break
    except Exception:
        pass
    return out

if __name__=="__main__":
    # demo with AAPL CIK 0000320193
    data = fetch_company_facts("0000320193")
    print(f"Fetched {len(str(data))} chars")
    print(parse_financials_from_facts(data))
