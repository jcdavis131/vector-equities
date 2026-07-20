"""
Feature spec for Vector Equities — 17 families ~128 features mirroring vector-hoops.
"""

FEATURE_FAMILIES = {
    "income": [
        "REV",
        "COGS",
        "GROSS_PROFIT",
        "OP_INCOME",
        "EBITDA",
        "NET_INCOME",
        "EBIT",
        "GROSS_MARGIN",
        "OP_MARGIN",
        "NET_MARGIN",
        "EBITDA_MARGIN",
    ],
    "balance": [
        "TOTAL_ASSETS",
        "TOTAL_LIABILITIES",
        "EQUITY",
        "CASH",
        "DEBT",
        "BOOK_VALUE",
        "TANGIBLE_BOOK",
        "WORKING_CAPITAL",
        "NET_DEBT",
        "INVESTED_CAPITAL",
    ],
    "cashflow": [
        "OCF",
        "CAPEX",
        "FCF",
        "FCF_MARGIN",
        "OCF_TO_NET",
        "FCF_CONVERSION",
        "CAPEX_TO_REV",
    ],
    "growth": [
        "REV_YOY",
        "EBITDA_YOY",
        "NET_YOY",
        "FCF_YOY",
        "REV_3Y_CAGR",
        "EBITDA_3Y_CAGR",
        "EPS_3Y_CAGR",
        "BOOK_3Y_CAGR",
        "OCF_3Y_CAGR",
    ],
    "profitability": [
        "ROE",
        "ROA",
        "ROIC",
        "GROSS_MARGIN",
        "OP_MARGIN",
        "NET_MARGIN",
        "FCF_ROIC",
        "EBITDA_MARGIN",
        "ROIC_WACC_SPREAD",
    ],
    "leverage_liquidity": [
        "CURRENT_RATIO",
        "QUICK_RATIO",
        "DEBT_TO_EQUITY",
        "DEBT_TO_EBITDA",
        "INTEREST_COVERAGE",
        "DEBT_TO_ASSETS",
        "NET_DEBT_TO_EBITDA",
    ],
    "efficiency": [
        "ASSET_TURNOVER",
        "INVENTORY_TURNOVER",
        "RECEIVABLE_TURNOVER",
        "CASH_CONVERSION_CYCLE",
        "CAPEX_TO_DEPRE",
    ],
    "per_share": ["EPS_DILUTED", "BVPS", "FCFPS", "SHARES_YOY", "DILUTION_3Y"],
    "market_price": [
        "RET_1M",
        "RET_3M",
        "RET_6M",
        "RET_12M",
        "VOL_30D",
        "VOL_90D",
        "VOL_252D",
        "BETA_1Y",
        "VOLUME_AVG_30D",
        "MOMENTUM_12_1",
    ],
    "valuation": [
        "PE",
        "PB",
        "PS",
        "EV_EBITDA",
        "EV_SALES",
        "EARNINGS_YIELD",
        "FCF_YIELD",
        "DIV_YIELD",
    ],
    "management_neo": [
        "NEO_COUNT",
        "CEO_AGE",
        "CEO_TENURE",
        "CEO_FOUNDER_FLAG",
        "CEO_TOTAL_COMP",
        "CEO_EQUITY_PCT",
        "AVG_NEO_COMP",
        "CEO_PAY_RATIO",
        "BOARD_INDEP_PCT",
        "BOARD_SIZE",
        "INSIDER_OWN_PCT",
        "CEO_PAY_VS_SECTOR",
        "NEO_TURNOVER",
        "CEO_DUALITY",
    ],
    "ownership": [
        "INST_PCT",
        "INST_DELTA_QOQ",
        "INSIDER_NET_12M",
        "FLOAT_PCT",
        "TOP10_INST_CONC",
        "SHORT_INTEREST_PCT",
    ],
    "disclosure_text": [
        "MDA_LENGTH",
        "MDA_SENTIMENT",
        "RISK_FACTOR_COUNT",
        "RISK_CHANGE_YOY",
        "FOG_INDEX_PROXY",
        "TONE_UNCERTAINTY",
    ],
    "sector_context": ["SECTOR_REL_RET_12M", "SECTOR_CONCENTRATION", "SECTOR_BETA"],
    "macro_regime": ["RATE_10Y", "VIX_AVG_FY", "CREDIT_SPREAD_PROXY", "GDP_GROWTH_FY"],
    "form": [
        "EARN_SURPRISE_STREAK",
        "GUIDANCE_RAISE_FLAG",
        "EPS_REVISION_UP_PCT",
        "PRICE_VS_52W_HIGH",
        "RSI_14_PROXY",
        "ACCIDENT_DISCLOSURE",
    ],
    "bbref_bridge": ["ALTMAN_Z", "PIOTROSKI_F_SCORE_PROXY"],
}

# flatten
ALL_FEATURES = []
FAMILY_OF = {}
for fam, feats in FEATURE_FAMILIES.items():
    for f in feats:
        ALL_FEATURES.append(f)
        FAMILY_OF[f] = fam

# 12 skills (Financial Crafts Lens)
SKILL_KEYS = [
    "Profitability",
    "Growth",
    "Moat_Margin_Stability",
    "Cash_Conversion",
    "Capital_Allocation",
    "Balance_Health",
    "Efficiency",
    "Valuation_Discipline",
    "Market_Momentum",
    "Management_Quality",
    "Shareholder_Yield",
    "Disclosure_Quality",
]

# 8 archetypes auto-named (k-means centroids will be labeled)
ARCHETYPE_NAMES = [
    "Compounder",
    "Cash_Cow",
    "Turnaround",
    "HyperGrowth_SaaS",
    "Heavy_Industrial",
    "Bank_Capital_Heavy",
    "Moonshot_Bio",
    "Serial_Acquirer",
]

# 11 GICS sectors
SECTORS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer_Discretionary",
    "Consumer_Staples",
    "Industrials",
    "Energy",
    "Materials",
    "Utilities",
    "Real_Estate",
    "Communication",
]

# Game profile 14-d equivalent for equities (core interpretable)
GAME_PROFILE_FEATURES = [
    "REV_YOY",
    "NET_MARGIN",
    "ROE",
    "ROIC",
    "FCF_MARGIN",
    "DEBT_TO_EBITDA",
    "CURRENT_RATIO",
    "ASSET_TURNOVER",
    "RET_12M",
    "EV_EBITDA",
    "PE",
    "CEO_TOTAL_COMP",
    "INSIDER_OWN_PCT",
    "ALTMAN_Z",
]
