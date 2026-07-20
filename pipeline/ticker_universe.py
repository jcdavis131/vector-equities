"""
Ticker Universe — robust version using company_tickers.json + fallback CSV handling
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "pipeline" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

USER_AGENT = "VectorEquities research (contact via GitHub)"

SP500_FALLBACK = [
    ("AAPL", "Technology"),
    ("MSFT", "Technology"),
    ("NVDA", "Technology"),
    ("AMZN", "Consumer Discretionary"),
    ("META", "Communication Services"),
    ("GOOGL", "Communication Services"),
    ("BRK.B", "Financials"),
    ("LLY", "Healthcare"),
    ("AVGO", "Technology"),
    ("JPM", "Financials"),
    ("UNH", "Healthcare"),
    ("V", "Financials"),
    ("MA", "Financials"),
    ("HD", "Consumer Discretionary"),
    ("COST", "Consumer Staples"),
    ("XOM", "Energy"),
    ("PG", "Consumer Staples"),
    ("CRM", "Technology"),
    ("WMT", "Consumer Staples"),
    ("NFLX", "Communication Services"),
    ("ORCL", "Technology"),
    ("CVX", "Energy"),
    ("KO", "Consumer Staples"),
    ("ABBV", "Healthcare"),
    ("MRK", "Healthcare"),
    ("ADBE", "Technology"),
    ("PEP", "Consumer Staples"),
    ("BAC", "Financials"),
    ("ACN", "Technology"),
    ("MCD", "Consumer Discretionary"),
    ("TMO", "Healthcare"),
    ("CSCO", "Technology"),
    ("LIN", "Materials"),
    ("ABT", "Healthcare"),
    ("WFC", "Financials"),
    ("AMD", "Technology"),
    ("DIS", "Communication Services"),
    ("GE", "Industrials"),
    ("DHR", "Healthcare"),
    ("VZ", "Communication Services"),
    ("CMCSA", "Communication Services"),
    ("PM", "Consumer Staples"),
    ("TXN", "Technology"),
    ("NEE", "Utilities"),
    ("QCOM", "Technology"),
    ("RTX", "Industrials"),
    ("T", "Communication Services"),
    ("IBM", "Technology"),
    ("AMGN", "Healthcare"),
    ("HON", "Industrials"),
    ("CAT", "Industrials"),
    ("SPG", "Real Estate"),
    ("BA", "Industrials"),
    ("INTU", "Technology"),
    ("MS", "Financials"),
    ("GS", "Financials"),
    ("BLK", "Financials"),
    ("PFE", "Healthcare"),
    ("LOW", "Consumer Discretionary"),
    ("BKNG", "Consumer Discretionary"),
    ("ISRG", "Healthcare"),
    ("GILD", "Healthcare"),
    ("DE", "Industrials"),
    ("AXP", "Financials"),
    ("SYK", "Healthcare"),
    ("CB", "Financials"),
    ("ADP", "Technology"),
    ("MMC", "Financials"),
    ("REGN", "Healthcare"),
    ("SBUX", "Consumer Discretionary"),
    ("PLD", "Real Estate"),
    ("LMT", "Industrials"),
    ("ADI", "Technology"),
    ("PANW", "Technology"),
    ("LRCX", "Technology"),
    ("ZTS", "Healthcare"),
    ("AMAT", "Technology"),
    ("CI", "Healthcare"),
    ("AMT", "Real Estate"),
    ("MDLZ", "Consumer Staples"),
    ("SCHW", "Financials"),
    ("PYPL", "Financials"),
    ("SNPS", "Technology"),
    ("HCA", "Healthcare"),
    ("ETN", "Industrials"),
    ("KLAC", "Technology"),
    ("BSX", "Healthcare"),
    ("SHW", "Materials"),
    ("MU", "Technology"),
    ("EL", "Consumer Staples"),
    ("WM", "Industrials"),
    ("ICE", "Financials"),
    ("MO", "Consumer Staples"),
    ("NKE", "Consumer Discretionary"),
]

SECTOR_MAP = {
    "Information Technology": "Technology",
    "Technology": "Technology",
    "Health Care": "Healthcare",
    "Healthcare": "Healthcare",
    "Financials": "Financials",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "ConsStaples": "Consumer Staples",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Materials": "Materials",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication",
    "Communication": "Communication",
}


def robust_fetch(url):
    import http.client
    import urllib.request

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except http.client.IncompleteRead as e:
            print(
                f"IncompleteRead for {url}: got {len(e.partial)} bytes, using partial"
            )
            return e.partial
        except Exception as e:
            print(f"Fetch attempt {attempt} fail {url}: {e}")
            time.sleep(1 + attempt)
    return None


def fetch_sec_tickers():
    cache = CACHE / "company_tickers.json"
    if cache.exists() and cache.stat().st_size > 100000:
        try:
            data = json.loads(cache.read_text())
            print(f"Loaded cached SEC {len(data)} tickers")
            return data
        except:
            pass
    for url in [
        "https://www.sec.gov/files/company_tickers.json",
        "https://www.sec.gov/files/company_tickers_exchange.json",
    ]:
        print(f"Fetching SEC {url}")
        raw = robust_fetch(url)
        if not raw:
            continue
        try:
            j = json.loads(raw.decode())
            cache.write_bytes(raw[:20000000])
            print(f"Parsed {len(j)} entries from {url}")
            return j
        except Exception as e:
            print(f"Parse fail {url}: {e}")
            continue
    return {}


def fetch_sp500_list():
    import csv
    import io

    cache = CACHE / "sp500.csv"
    # Prefer local cache if it looks valid (>50 lines, has META)
    if cache.exists() and cache.stat().st_size > 5000:
        try:
            txt = cache.read_text()
            if "META" in txt and "Symbol" in txt:
                reader = csv.DictReader(io.StringIO(txt))
                out = []
                for row in reader:
                    sym = (row.get("Symbol") or row.get("Ticker") or "").strip()
                    name = (row.get("Security") or row.get("Name") or sym).strip()
                    sec = (row.get("GICS Sector") or row.get("Sector") or "Industrials").strip()
                    if sym:
                        out.append((sym, name, sec))
                if len(out) >= 400:
                    print(f"Loaded cached S&P {len(out)} from {cache}")
                    return out
        except Exception as e:
            print(f"Cache S&P parse fail: {e}")

    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv",
    ]
    for url in urls:
        raw = robust_fetch(url)
        if not raw:
            continue
        try:
            txt = raw.decode()
            if "Symbol" in txt or "Ticker" in txt or "AAPL" in txt:
                # Validate with csv parsing
                reader = csv.DictReader(io.StringIO(txt))
                out = []
                for row in reader:
                    sym = (row.get("Symbol") or row.get("Ticker") or "").strip()
                    name = (row.get("Security") or row.get("Name") or sym).strip()
                    sec = (row.get("GICS Sector") or row.get("Sector") or "Industrials").strip()
                    if sym:
                        out.append((sym, name, sec))
                if len(out) > 200:  # must be substantial
                    cache.write_bytes(raw)
                    print(f"Got {len(out)} S&P from {url}")
                    return out
        except Exception as e:
            print(f"S&P parse fail {url}: {e}")
    print("Using fallback SP500 list")
    return [(t, t, s) for t, s in SP500_FALLBACK]


def build_universe(limit=500):
    sec_data = fetch_sec_tickers()
    ticker_to_cik = {}
    if isinstance(sec_data, dict):
        for v in sec_data.values():
            try:
                ticker = str(v.get("ticker", "")).strip().upper()
                cik = str(v.get("cik_str", "")).zfill(10)
                if ticker:
                    ticker_to_cik[ticker] = {
                        "cik": cik,
                        "title": v.get("title", ""),
                        "exchange": v.get("exchange", ""),
                    }
            except:
                continue
    print(f"SEC map {len(ticker_to_cik)} tickers")

    sp = fetch_sp500_list()
    universe = []
    for sym, name, raw_sector in sp[: limit * 2]:
        ticker = sym.strip().upper()
        ticker_yf = ticker.replace(".", "-")
        info = None
        for cand in [
            ticker,
            ticker.replace("-", "."),
            ticker.replace(".", "-"),
            ticker.split(".")[0],
        ]:
            if cand in ticker_to_cik:
                info = ticker_to_cik[cand]
                break
        if not info:
            continue
        mapped_sector = SECTOR_MAP.get(raw_sector.strip(), raw_sector.strip())
        universe.append(
            {
                "ticker": ticker_yf,
                "ticker_sec": ticker,
                "cik": info["cik"],
                "company": name,
                "sector": mapped_sector,
                "sector_raw": raw_sector,
                "exchange": info.get("exchange", ""),
            }
        )
        if len(universe) >= limit:
            break

    if len(universe) < 100:
        print(f"Only {len(universe)} mapped, topping up with fallback")
        for t, s in SP500_FALLBACK:
            if len(universe) >= limit:
                break
            if any(u["ticker"].startswith(t.split(".")[0]) for u in universe):
                continue
            info = ticker_to_cik.get(t) or ticker_to_cik.get(t.replace(".", "-"))
            if not info:
                base = t.split(".")[0]
                info = ticker_to_cik.get(base)
            if not info:
                continue
            universe.append(
                {
                    "ticker": t.replace(".", "-"),
                    "ticker_sec": t,
                    "cik": info["cik"],
                    "company": info["title"],
                    "sector": SECTOR_MAP.get(s, s),
                    "sector_raw": s,
                    "exchange": info.get("exchange", ""),
                }
            )

    out_path = DATA_DIR / "universe.json"
    out_path.write_text(json.dumps(universe, indent=2))
    print(f"Saved {len(universe)} universe to {out_path}")
    return universe


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    build_universe(limit=args.limit)
