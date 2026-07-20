"""
Fetch external free datasets for v6 towers
- GPR, EPU, GSCPI, election calendar
All cached to pipeline/data/external/
"""

from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path("pipeline/data/external")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_gpr():
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    out = DATA_DIR / "gpr_export.xls"
    try:
        print(f"Fetching GPR from {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"Saved GPR to {out} {len(r.content)} bytes")
    except Exception as e:
        print(f"GPR fetch failed {e}, will use fallback synthetic")


def fetch_epu():
    # EPU daily for US
    urls = [
        "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv",
        "https://www.policyuncertainty.com/media/EPU_Data.csv",
    ]
    for url in urls:
        try:
            out = DATA_DIR / f"epu_{Path(url).name}"
            print(f"Fetching EPU {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            out.write_bytes(r.content)
            print(f"Saved {out}")
            break
        except Exception as e:
            print(f"EPU fetch {url} failed {e}")


def fetch_gscpi():
    url = "https://www.newyorkfed.org/medialibrary/research/gscpi/gscpi_data.xlsx"
    out = DATA_DIR / "gscpi_data.xlsx"
    try:
        print(f"Fetching GSCPI {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"Saved GSCPI {out}")
    except Exception as e:
        print(f"GSCPI fetch failed {e}")


def fetch_commodities_yfinance():
    # Download 2014-2025 monthly for commodities to compute FY avg YoY
    tickers = [
        "CL=F",
        "BZ=F",
        "HG=F",
        "SLX",
        "LBS=F",
        "NG=F",
        "DX-Y.NYB",
        "CNY=X",
        "BDRY",
        "C=F",
        "W=F",
    ]
    # Use yfinance batch
    try:
        print("Fetching commodities via yfinance")
        data = yf.download(
            tickers,
            start="2014-01-01",
            end="2025-07-17",
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )
        # data is multi
        out = DATA_DIR / "commodities_monthly.csv"
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"] if "Close" in data else data
        else:
            close = data
        close.to_csv(out)
        print(f"Saved commodities {out}")
    except Exception as e:
        print(f"commodities yfinance failed {e}")


if __name__ == "__main__":
    fetch_gpr()
    fetch_epu()
    fetch_gscpi()
    fetch_commodities_yfinance()
