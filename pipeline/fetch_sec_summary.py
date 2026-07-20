"""Fetch SEC facts and save tiny yearly summary only — memory efficient"""

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from fetch_sec import CACHE, robust_fetch_json

CACHE_SUM = CACHE / "sec_summary"
CACHE_SUM.mkdir(parents=True, exist_ok=True)

TAGS = {
    "REVENUE": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueNetOrGrossProfit",
    ],
    "COGS": ["CostOfGoodsSold", "CostOfGoodsAndServicesSold"],
    "GROSS": ["GrossProfit"],
    "OP_INCOME": ["OperatingIncomeLoss"],
    "NET_INCOME": ["NetIncomeLoss"],
    "ASSETS": ["Assets"],
    "LIAB": ["Liabilities", "LiabilitiesCurrent"],
    "EQUITY": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "CASH": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents"],
    "DEBT_LT": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "DEBT_ST": ["ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent"],
    "OCF": ["NetCashProvidedByUsedInOperatingActivities", "OperatingCashFlow"],
    "CAPEX": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
        "PurchasesOfPropertyPlantAndEquipment",
    ],
    "CURR_A": ["AssetsCurrent"],
    "CURR_L": ["LiabilitiesCurrent"],
    "DEPR": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "INTEREST": ["InterestExpense", "InterestExpenseDebt"],
    "SHARES_D": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
        "CommonStockSharesOutstanding",
    ],
    "SHARES_B": ["WeightedAverageNumberOfSharesOutstandingBasic"],
    "RET_EARN": ["RetainedEarningsAccumulatedDeficit"],
}


def get_fact_for_year_fast(facts, tag_names, year):
    # fast path: look in us-gaap
    try:
        us = facts.get("facts", {}).get("us-gaap", {})
        for tag in tag_names:
            if tag not in us:
                continue
            units = us[tag].get("units", {})
            for unit_key in ("USD", "USD/shares", "shares"):
                if unit_key not in units:
                    continue
                # filter by FY and frame CYxxxx or FY+year
                candidates = []
                for entry in units[unit_key]:
                    # entry has end date and fy, fp, form, frame
                    # frame like CY2020 or CY2020Q4 etc, or CY2020I?
                    frame = entry.get("frame", "")
                    fy = entry.get("fy")
                    fp = entry.get("fp", "")
                    # match year
                    if fy == year and fp == "FY":
                        candidates.append(entry)
                    elif frame.startswith(f"CY{year}") and (
                        "Q4" in frame or len(frame) == 6
                    ):
                        candidates.append(entry)
                    elif frame == f"CY{year}":
                        candidates.append(entry)
                if candidates:
                    # pick latest filed?
                    candidates.sort(key=lambda x: x.get("filed", ""))
                    best = candidates[-1]
                    return best.get("val")
                # fallback: any with end date in year
                for entry in units[unit_key]:
                    end = entry.get("end", "")
                    if str(year) in end and entry.get("fp") == "FY":
                        return entry.get("val")
        return None
    except Exception:
        return None


def fetch_summary_for_cik(cik: str):
    cik_pad = str(cik).zfill(10)
    out_path = CACHE_SUM / f"summary_{cik_pad}.json"
    if out_path.exists() and out_path.stat().st_size > 100:
        try:
            return json.loads(out_path.read_text())
        except:
            out_path.unlink(missing_ok=True)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_pad}.json"
    data = robust_fetch_json(url)
    if not data:
        return None
    # extract per year 2015-2024
    summary = {}
    for yr in range(2015, 2025):
        yearly = {}
        for key, tags in TAGS.items():
            val = get_fact_for_year_fast(data, tags, yr)
            if val is not None:
                yearly[key] = val
        summary[str(yr)] = yearly
    summary["_meta"] = {
        "cik": cik_pad,
        "entity": data.get("entityName"),
        "fetched": time.time(),
    }
    # save
    try:
        out_path.write_text(json.dumps(summary))
    except:
        pass
    # free
    del data
    gc.collect()
    time.sleep(0.25)
    return summary


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "pipeline" / "data"
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    subset = uni[args.start : args.start + args.limit]
    print(f"Fetching summaries {len(subset)} from {args.start}")
    ok = 0
    for i, entry in enumerate(subset):
        cik = entry["cik"]
        ticker = entry["ticker"]
        print(f"[{i + 1}/{len(subset)}] {ticker} CIK {cik}", flush=True)
        s = fetch_summary_for_cik(cik)
        if s:
            ok += 1
            # print yearly revs
            revs = [
                f"{yr}:{s.get(str(yr), {}).get('REVENUE', '-')}"
                for yr in range(2020, 2025)
            ]
            # print count
            cnt = sum(
                1
                for yr in range(2015, 2025)
                if s.get(str(yr), {}).get("REVENUE") is not None
            )
            print(f"  -> ok rev years {cnt} {revs[-1]}")
        else:
            print("  -> fail")
    print(f"Done {ok}/{len(subset)}")
