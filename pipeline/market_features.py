"""Real market-derived features from pipeline/cache/market_history/{ticker}.json.

Replaces the market_price + part of the valuation family in
build_real_from_summary.py, which previously hardcoded every one of these to
None (RET_1M..MOMENTUM_12_1, PE/PB/PS/EV_*) -- 15 features, 0% coverage,
despite fetch_market_history.py already pulling real 10y daily OHLCV for
every ticker in the universe. This was simply never joined in.

Approximates each fiscal year's "as of" date as Dec 31 of that year (the rest
of build_real_from_summary.py already treats fiscal year as a plain calendar
year everywhere, so this matches its own precision level -- not introducing
a new approximation, just being consistent with the existing one).

Coverage is honest: a ticker/year with no market_history cache entry, or with
too few trading days before the as-of date for a given window, returns None
for that specific field -- never imputed.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "market_history"

_ticker_cache: dict[str, dict] = {}


def _load(ticker: str) -> dict | None:
    if ticker in _ticker_cache:
        return _ticker_cache[ticker]
    p = CACHE / f"{ticker}.json"
    if not p.exists():
        _ticker_cache[ticker] = None
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        hist = d.get("history", [])
        hist.sort(key=lambda r: r["date"])
        _ticker_cache[ticker] = hist
        return hist
    except Exception:
        _ticker_cache[ticker] = None
        return None


def _rows_on_or_before(hist: list[dict], as_of: str) -> list[dict]:
    return [r for r in hist if r["date"] <= as_of]


def _returns(closes: list[float]) -> list[float]:
    out = []
    for i in range(1, len(closes)):
        if closes[i - 1]:
            out.append((closes[i] - closes[i - 1]) / closes[i - 1])
    return out


def _price_n_days_before(rows: list[dict], n_trading_days: int) -> float | None:
    if len(rows) <= n_trading_days:
        return None
    return rows[-1 - n_trading_days]["close"]


def get_market_row(ticker: str, year: int) -> dict:
    """Real market/valuation features as of Dec 31 of `year`. None where the
    ticker has no cached history or too few trading days for a given window.
    """
    out = {
        "RET_1M": None, "RET_3M": None, "RET_6M": None, "RET_12M": None,
        "VOL_30D": None, "VOL_90D": None, "VOL_252D": None, "BETA_1Y": None,
        "VOLUME_AVG_30D": None, "MOMENTUM_12_1": None,
        "PRICE_VS_52W_HIGH": None, "RSI_14_PROXY": None,
        "_price": None, "_shares_implied_ok": False,
    }
    hist = _load(ticker)
    if not hist:
        return out
    as_of = f"{year}-12-31"
    rows = _rows_on_or_before(hist, as_of)
    if len(rows) < 22:
        return out
    price = rows[-1]["close"]
    out["_price"] = price

    for months, n_days, key in ((1, 21, "RET_1M"), (3, 63, "RET_3M"),
                                 (6, 126, "RET_6M"), (12, 252, "RET_12M")):
        p0 = _price_n_days_before(rows, n_days)
        if p0:
            out[key] = (price - p0) / p0

    closes = [r["close"] for r in rows]
    for n_days, key in ((30, "VOL_30D"), (90, "VOL_90D"), (252, "VOL_252D")):
        window = closes[-n_days:] if len(closes) >= n_days else None
        if window and len(window) >= max(10, n_days // 3):
            rets = _returns(window)
            if len(rets) >= 5:
                mu = sum(rets) / len(rets)
                var = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
                out[key] = math.sqrt(var) * math.sqrt(252)  # annualized

    if len(rows) >= 252:
        vols = [r["volume"] for r in rows[-30:]]
        if vols:
            out["VOLUME_AVG_30D"] = sum(vols) / len(vols)

    p12 = _price_n_days_before(rows, 252)
    p1 = _price_n_days_before(rows, 21)
    if p12 and p1:
        out["MOMENTUM_12_1"] = (p1 - p12) / p12

    window_52w = rows[-252:] if len(rows) >= 252 else rows
    if window_52w:
        high_52w = max(r["high"] for r in window_52w)
        if high_52w:
            out["PRICE_VS_52W_HIGH"] = price / high_52w

    if len(closes) >= 15:
        window14 = closes[-15:]
        deltas = [window14[i] - window14[i - 1] for i in range(1, len(window14))]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss == 0:
            out["RSI_14_PROXY"] = 100.0
        else:
            rs = avg_gain / avg_loss
            out["RSI_14_PROXY"] = 100.0 - (100.0 / (1.0 + rs))

    spy_hist = _load("SPY")
    if spy_hist and len(rows) >= 252:
        spy_rows = _rows_on_or_before(spy_hist, as_of)
        # align by date on the trailing ~252 trading days both series share
        stock_by_date = {r["date"]: r["close"] for r in rows[-260:]}
        spy_by_date = {r["date"]: r["close"] for r in spy_rows[-260:]}
        shared = sorted(set(stock_by_date) & set(spy_by_date))
        if len(shared) >= 60:
            s_closes = [stock_by_date[d] for d in shared]
            m_closes = [spy_by_date[d] for d in shared]
            s_ret = _returns(s_closes)
            m_ret = _returns(m_closes)
            n = min(len(s_ret), len(m_ret))
            if n >= 30:
                s_ret, m_ret = s_ret[-n:], m_ret[-n:]
                m_mu = sum(m_ret) / n
                s_mu = sum(s_ret) / n
                cov = sum((s_ret[i] - s_mu) * (m_ret[i] - m_mu) for i in range(n)) / (n - 1)
                var_m = sum((r - m_mu) ** 2 for r in m_ret) / (n - 1)
                if var_m > 1e-12:
                    out["BETA_1Y"] = cov / var_m
    return out


def valuation_row(price: float | None, shares: float | None, eps: float | None,
                   bvps: float | None, rev: float | None, ebitda: float | None,
                   fcf: float | None, debt: float | None, cash: float | None,
                   dividends_paid: float | None) -> dict:
    """PE/PB/PS/EV_*/yields from real price + fundamentals already computed
    upstream (shares/eps/bvps/rev/ebitda/fcf/debt/cash are all real SEC-
    derived values already flowing through build_real_from_summary.py)."""
    out = {"PE": None, "PB": None, "PS": None, "EV_EBITDA": None, "EV_SALES": None,
           "EARNINGS_YIELD": None, "FCF_YIELD": None, "DIV_YIELD": None}
    if price is None or shares is None or shares <= 0:
        return out
    mkt_cap = price * shares
    if eps and eps > 0:
        out["PE"] = price / eps
        out["EARNINGS_YIELD"] = eps / price
    if bvps and bvps > 0:
        out["PB"] = price / bvps
    if rev and rev > 0:
        out["PS"] = mkt_cap / rev
        ev = mkt_cap + (debt or 0) - (cash or 0)
        out["EV_SALES"] = ev / rev
    if ebitda and ebitda > 0:
        ev = mkt_cap + (debt or 0) - (cash or 0)
        out["EV_EBITDA"] = ev / ebitda
    if fcf and mkt_cap > 0:
        out["FCF_YIELD"] = fcf / mkt_cap
    if dividends_paid and dividends_paid > 0:
        out["DIV_YIELD"] = dividends_paid / mkt_cap
    return out
