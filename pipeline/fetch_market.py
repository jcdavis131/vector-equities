"""
Market data fetcher via yfinance (free tier) — price, returns, vol, beta
Cache to pipeline/cache/market

"""
from pathlib import Path
import json, time
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "market"
CACHE.mkdir(parents=True, exist_ok=True)

def fetch_ticker_history(ticker: str, period="5y"):
    try:
        import yfinance as yf
        cache_file = CACHE / f"{ticker}_{period}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty:
            return None
        # compute features
        hist["RET_1D"] = hist["Close"].pct_change()
        out = {
            "ticker": ticker,
            "last_close": float(hist["Close"].iloc[-1]),
            "avg_vol_30d": float(hist["Volume"].tail(30).mean()),
            "ret_12m": float(hist["Close"].pct_change(252).iloc[-1]) if len(hist)>252 else 0,
            "vol_252d": float(hist["RET_1D"].tail(252).std()* (252**0.5)) if len(hist)>30 else 0.3,
            "price_vs_52w": float(hist["Close"].iloc[-1] / hist["Close"].tail(252).max()) if len(hist)>252 else 0.9,
        }
        cache_file.write_text(json.dumps(out, indent=2))
        time.sleep(0.3)  # be nice
        return out
    except Exception as e:
        print(f"yfinance fail {ticker}: {e} — using synthetic fallback")
        return None

if __name__=="__main__":
    print(fetch_ticker_history("AAPL"))
