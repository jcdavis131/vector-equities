"""
DEF 14A Named Executive Officer parser — extracts NEO features for management tower

Real parser would use SEC 14A HTML + Regex. This scaffold demonstrates fields.

"""
from pathlib import Path
import re, json
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "proxy"
CACHE.mkdir(parents=True, exist_ok=True)

NEO_PATTERN = {
    "neo_count": r"Named Executive Officers.*?(\d+)",
    "ceo_tenure": r"Chief Executive.*?(\d+)\s*years",
}

def parse_proxy_text(text: str) -> dict:
    """Extract NEO features from DEF 14A text (simplified)"""
    out = {
        "NEO_COUNT": 5,
        "CEO_AGE": 55,
        "CEO_TENURE": 6.0,
        "CEO_FOUNDER_FLAG": 0.0,
        "CEO_TOTAL_COMP": 12.0,  # log scale placeholder
        "CEO_EQUITY_PCT": 2.5,
        "AVG_NEO_COMP": 11.0,
        "CEO_PAY_RATIO": 200.0,
        "BOARD_INDEP_PCT": 75.0,
        "BOARD_SIZE": 9.0,
        "INSIDER_OWN_PCT": 3.0,
    }
    # naive regex augment
    try:
        # founder mentions
        if re.search(r"founder", text, re.I):
            out["CEO_FOUNDER_FLAG"]=1.0
        # age extract
        m=re.search(r"Age\s*(\d{2})", text)
        if m:
            out["CEO_AGE"]=float(m.group(1))
    except:
        pass
    return out

def fetch_and_parse_def14a(cik: str):
    # placeholder — real would fetch from EDGAR def14a filings list
    # for now return synthetic
    return parse_proxy_text("Sample proxy statement with Named Executive Officers")

if __name__=="__main__":
    print(fetch_and_parse_def14a("0000320193"))
