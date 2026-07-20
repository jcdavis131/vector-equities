"""
Global trade & commodity tower via yfinance + GSCPI + synthetic fallback
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("pipeline/data/external")


def synthetic_trade(years=range(2015, 2025)):
    np.random.seed(456)
    rows = []
    for year in years:
        oil_yoy = np.random.normal(0.1, 0.4) + (
            -0.5 if year == 2020 else 0.6 if year == 2022 else 0
        )
        brent_spread = np.random.normal(2, 1)
        copper_yoy = np.random.normal(0.05, 0.3) + (
            0.4 if year == 2021 else -0.2 if year == 2020 else 0
        )
        steel_yoy = np.random.normal(0.08, 0.35)
        lumber_yoy = np.random.normal(0.0, 0.5) + (
            1.0 if year == 2021 else -0.6 if year == 2022 else 0
        )
        natgas_yoy = np.random.normal(0.2, 0.6) + (1.2 if year == 2022 else 0)
        dxy_yoy = np.random.normal(0.02, 0.08) + (
            0.1 if year == 2022 else -0.05 if year == 2023 else 0
        )
        usdcny_yoy = np.random.normal(0.02, 0.06) + (
            0.08 if year in [2018, 2019] else 0
        )
        bdry_yoy = np.random.normal(0.1, 0.8) + (
            -0.6 if year == 2020 else 1.0 if year == 2021 else 0
        )
        gscpi = np.random.normal(0, 1) + (
            2.5 if year in [2021, 2022] else -0.5 if year == 2023 else 0
        )
        commodity_beta = np.random.uniform(
            -0.5, 1.5
        )  # placeholder sector interaction later
        agri_yoy = np.random.normal(0.05, 0.3) + (0.5 if year == 2022 else 0)
        rows.append(
            {
                "year": year,
                "OIL_WTI_YOY": oil_yoy,
                "OIL_BRENT_SPREAD": brent_spread,
                "COPPER_YOY": copper_yoy,
                "STEEL_PROXY_YOY": steel_yoy,
                "LUMBER_YOY": lumber_yoy,
                "NATGAS_YOY": natgas_yoy,
                "DXY_YOY": dxy_yoy,
                "USDCNY_YOY": usdcny_yoy,
                "BDRY_YOY": bdry_yoy,
                "GSCPI_AVG_FY": gscpi,
                "COMMODITY_BETA_X_SECTOR": commodity_beta,
                "AGRI_YOY": agri_yoy,
            }
        )
    df = pd.DataFrame(rows)
    # z-score except spread
    for col in df.columns:
        if col != "year" and "SPREAD" not in col:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
    return df


if __name__ == "__main__":
    df = synthetic_trade()
    df.to_csv(DATA_DIR / "trade_commodity_tower.csv", index=False)
    print(df)
