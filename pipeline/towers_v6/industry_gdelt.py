"""
Industry event tower via GDELT + fallback synthetic
Generates per-sector per-year features
"""

import numpy as np
import pandas as pd

SECTORS = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Healthcare",
    "Financials",
    "Technology",
    "Communication",
    "Utilities",
    "Real Estate",
]

# Map sector to GDELT query keywords (simplified)
SECTOR_QUERY = {
    "Energy": "energy oil gas",
    "Materials": "materials mining metals chemicals",
    "Industrials": "industrials manufacturing aerospace defense",
    "Technology": "technology software semiconductor",
    "Healthcare": "healthcare pharma biotech",
    "Financials": "banks finance fintech",
    "Consumer Discretionary": "retail consumer automotive",
    "Consumer Staples": "consumer staples food beverage",
    "Communication": "telecom media entertainment",
    "Utilities": "utilities electricity power",
    "Real Estate": "real estate REIT property",
}


def synthetic_industry_features(years=range(2015, 2025)):
    """Fallback synthetic that respects sector priors, for offline development"""
    np.random.seed(42)
    rows = []
    for sector in SECTORS:
        base_vol = np.random.uniform(0.5, 1.5)
        for year in years:
            # Create correlated features
            news_vol = base_vol + 0.3 * np.sin(year * 0.5) + np.random.normal(0, 0.2)
            tone = np.random.normal(0, 1.5) + (
                0.2 if sector == "Technology" else -0.1 if sector == "Energy" else 0
            )
            neg = max(
                0,
                int(
                    np.random.poisson(
                        3 + (2 if sector in ["Energy", "Materials"] else 0)
                    )
                ),
            )
            pos = int(np.random.poisson(4 + (1 if sector == "Technology" else 0)))
            reg = int(
                np.random.poisson(
                    2
                    + (2 if sector in ["Financials", "Healthcare", "Technology"] else 0)
                )
            )
            ma = int(np.random.poisson(3))
            supply = int(np.random.poisson(2 + (3 if year >= 2020 else 0)))
            # Derived
            earn_breadth = np.random.normal(0.5, 0.15)
            mom_disp = np.random.uniform(0.1, 0.4)
            vol_spike = np.random.uniform(0.8, 1.5) + (
                0.3 if year in [2020, 2022] else 0
            )
            rows.append(
                {
                    "sector": sector,
                    "year": year,
                    "IND_NEWS_VOL_Z": news_vol,
                    "IND_NEWS_TONE_AVG": tone,
                    "IND_NEG_EVENT_CNT": neg,
                    "IND_POS_EVENT_CNT": pos,
                    "IND_REGULATORY_RISK": reg,
                    "IND_MA_INTENSITY": ma,
                    "IND_SUPPLY_DISRUPTION": supply,
                    "IND_EARN_BREADTH": earn_breadth,
                    "IND_DISPERSION_MOM": mom_disp,
                    "IND_VOL_SPIKE": vol_spike,
                }
            )
    df = pd.DataFrame(rows)
    # Z-score per feature vs sector history
    for col in df.columns:
        if col not in ["sector", "year"]:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
    return df


def fetch_gdelt_live():
    """TODO: implement GDELT Doc API query - for now return synthetic"""
    # Real implementation would:
    # for each sector/year: query https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=timelinevol&...
    # Parse JSON
    # Rate limit 0.25s
    return synthetic_industry_features()


if __name__ == "__main__":
    df = synthetic_industry_features()
    df.to_csv("pipeline/data/external/industry_event_tower.csv", index=False)
    print(f"Generated industry tower {df.shape}")
    print(df.head())
