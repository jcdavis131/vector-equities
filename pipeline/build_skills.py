"""
Financial Craft Skills — 12 skills graded 0-100 (like vector-hoops skills lens)
Each is linear composite of z-scored features, percentile within fiscal year.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "pipeline" / "data" / "train_matrix.npz"
MANIFEST_PATH = ROOT / "pipeline" / "data" / "feature_manifest.json"

SKILL_DEFS = {
    "Profitability": ["ROE","ROA","ROIC","NET_MARGIN","GROSS_MARGIN","OP_MARGIN"],
    "Growth": ["REV_YOY","REV_3Y_CAGR","EBITDA_YOY","NET_YOY","EPS_3Y_CAGR"],
    "Moat_Margin_Stability": ["GROSS_MARGIN","OP_MARGIN","NET_MARGIN","ROIC","ROIC_WACC_SPREAD"],
    "Cash_Conversion": ["FCF_MARGIN","OCF_TO_NET","FCF_CONVERSION","CASH_CONVERSION_CYCLE"],
    "Capital_Allocation": ["ROIC","FCF_MARGIN","CAPEX_TO_REV","SHARES_YOY","BVPS"],
    "Balance_Health": ["CURRENT_RATIO","DEBT_TO_EQUITY","DEBT_TO_EBITDA","ALTMAN_Z","INTEREST_COVERAGE"],
    "Efficiency": ["ASSET_TURNOVER","INVENTORY_TURNOVER","RECEIVABLE_TURNOVER"],
    "Valuation_Discipline": ["PE","PB","EV_EBITDA","EARNINGS_YIELD","FCF_YIELD"],  # inverted later
    "Market_Momentum": ["RET_12M","RET_6M","MOMENTUM_12_1","RET_3M","VOL_252D"],
    "Management_Quality": ["CEO_TENURE","CEO_FOUNDER_FLAG","INSIDER_OWN_PCT","BOARD_INDEP_PCT","CEO_TOTAL_COMP"],
    "Shareholder_Yield": ["DIV_YIELD","FCF_YIELD","INSIDER_OWN_PCT","SHARES_YOY"],
    "Disclosure_Quality": ["MDA_SENTIMENT","RISK_FACTOR_COUNT","MDA_LENGTH","FOG_INDEX_PROXY"]
}

def build():
    npz = np.load(DATA_PATH, allow_pickle=False)
    Z = npz["Z"]
    fiscal_years = npz["fiscal_year"]
    names = npz["name"]
    tickers = npz["ticker"]
    manifest = json.loads(MANIFEST_PATH.read_text())
    feat_to_idx = {f:i for i,f in enumerate(manifest["features"])}

    keys = list(SKILL_DEFS.keys())
    G = np.zeros((len(Z), len(keys)), dtype=np.float32)

    for j, skill in enumerate(keys):
        feats = SKILL_DEFS[skill]
        cols = [feat_to_idx.get(f) for f in feats if feat_to_idx.get(f) is not None]
        if not cols:
            continue
        raw = Z[:, cols].mean(axis=1)
        # For valuation discipline, lower multiples = higher skill (invert)
        if skill=="Valuation_Discipline":
            raw = -raw
        # For volatility, lower vol = higher momentum skill component partially inverted
        G[:, j] = raw

    # Per fiscal year percentile -> 0-100 grade
    grades = np.zeros_like(G)
    for fy in sorted(set(str(x) for x in fiscal_years)):
        rows = np.where(fiscal_years.astype(str)==fy)[0]
        for j in range(len(keys)):
            vals = G[rows, j]
            order = vals.argsort()
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(rows))
            perc = ranks / max(len(rows)-1,1)
            grades[rows, j] = perc*100

    out = ROOT / "pipeline" / "data" / "skill_labels.npz"
    np.savez_compressed(out, grades=grades, keys=np.array(keys), name=names, season=fiscal_years, ticker=tickers)
    print(f"Saved {grades.shape} skill grades to {out}")

    # also save wide? same for now
    return grades

if __name__=="__main__":
    build()
