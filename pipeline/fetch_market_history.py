"""
Market history fetcher for continuous career model — 10y daily OHLCV per ticker via yfinance free-tier
Cache to pipeline/cache/market_history/{ticker}.json with full daily history + precomputed indicators

"""
from pathlib import Path
import json, time, datetime
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "market_history"
CACHE.mkdir(parents=True, exist_ok=True)
SPY_CACHE = CACHE / "SPY.json"
UNIVERSE = ROOT / "pipeline" / "data" / "universe.json"

def load_universe():
    try:
        data=json.loads(UNIVERSE.read_text())
        tickers = [d['ticker'] for d in data if 'ticker' in d]
        # deduplicate
        seen=set()
        uniq=[]
        for t in tickers:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq
    except:
        # fallback to market dir listing
        mdir = ROOT / "pipeline" / "cache" / "market"
        return [p.stem.split('_')[0] for p in mdir.glob("*.json")] if mdir.exists() else ["AAPL"]

def fetch_one(ticker, period="10y", force=False):
    out_file = CACHE / f"{ticker}.json"
    if out_file.exists() and not force:
        try:
            j=json.loads(out_file.read_text())
            if 'history' in j and len(j['history'])>500:
                return j
        except:
            pass
    try:
        import yfinance as yf
        import pandas as pd
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=False)  # keep OHLCV non-adjusted for vol? Use auto_adjust False for close vs adj?
        if hist.empty:
            print(f"{ticker} empty")
            return None
        hist = hist.reset_index()
        # ensure Date is string
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        # compute daily returns
        import numpy as np
        close = hist['Close'].values
        ret_1d = np.zeros_like(close)
        ret_1d[1:] = (close[1:]-close[:-1])/close[:-1]
        # store minimal
        records = []
        for i,row in hist.iterrows():
            records.append({
                "date": row['Date'],
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume']) if not pd.isna(row['Volume']) else 0.0,
            })
        # compute 52w high rolling etc later per FY, but precompute some summary
        last = records[-1]
        out={
            "ticker": ticker,
            "last_close": last['close'],
            "history": records,
            "fetched_at": datetime.datetime.utcnow().isoformat(),
            "count": len(records),
        }
        out_file.write_text(json.dumps(out))
        print(f"{ticker} fetched {len(records)} days last_close {last['close']:.2f}")
        time.sleep(0.25)
        return out
    except Exception as e:
        print(f"fail {ticker}: {e}")
        time.sleep(0.5)
        return None

def fetch_spy():
    return fetch_one("SPY", period="10y")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    args=ap.parse_args()
    tickers = args.tickers or load_universe()
    tickers = tickers[:args.limit]
    print(f"Fetching {len(tickers)} tickers history")
    # fetch SPY first for beta calc later
    if not SPY_CACHE.exists() or args.force:
        print("Fetching SPY")
        fetch_spy()
    # threaded fetch with rate limit via sequential sleep inside
    for i,t in enumerate(tickers):
        if (CACHE / f"{t}.json").exists() and not args.force:
            continue
        fetch_one(t, force=args.force)
        if i%20==0:
            print(f"progress {i}/{len(tickers)}")
