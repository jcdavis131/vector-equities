"""
Political risk tower: GPR, EPU, election proximity
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("pipeline/data/external")


def synthetic_political(years=range(2015, 2025)):
    np.random.seed(123)
    # US presidential elections: 2016, 2020, 2024
    rows = []
    for year in years:
        # GPR - higher in 2022 Russia-Ukraine, 2020, 2024
        gpr = (
            100
            + (50 if year == 2022 else 20 if year in [2020, 2024] else 0)
            + np.random.normal(0, 10)
        )
        gpr_yoy = gpr - (100 + np.random.normal(0, 5))
        epu_us = (
            100 + (80 if year in [2020, 2016, 2024] else 0) + np.random.normal(0, 15)
        )
        epu_global = epu_us * 0.8 + np.random.normal(0, 10)
        tariff = (
            50
            + (100 if year in [2018, 2019, 2024, 2025] else 0)
            + np.random.normal(0, 10)
        )
        # election proximity US: 1/(months to election)
        if year in [2016, 2020, 2024]:
            elec_us = 1.0
        elif year in [2015, 2019, 2023]:
            elec_us = 0.7
        else:
            elec_us = 0.2
        elec_global = np.random.uniform(0.3, 0.8) + (0.2 if year % 2 == 0 else 0)
        wgi = np.random.normal(0.5, 0.1)
        gov_shutdown = 1.0 if year in [2018, 2023] else np.random.uniform(0, 0.3)
        rate_vol = (
            0.5 + (1.5 if year in [2022, 2023] else 0.2) + np.random.normal(0, 0.2)
        )
        rows.append(
            {
                "year": year,
                "GPR_GLOBAL_AVG_FY": gpr,
                "GPR_YOY": gpr_yoy,
                "EPU_US_AVG_FY": epu_us,
                "EPU_GLOBAL_AVG_FY": epu_global,
                "ELEC_PROX_US": elec_us,
                "ELEC_PROX_GLOBAL": elec_global,
                "TARIFF_RISK": tariff,
                "WGI_POL_STABILITY": wgi,
                "GOV_SHUTDOWN_PROX": gov_shutdown,
                "RATE_VOL_3M": rate_vol,
            }
        )
    df = pd.DataFrame(rows)
    # z-score
    for col in df.columns:
        if col != "year":
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
    return df


def load_real_if_exists():
    # Try to load real GPR/EPU if fetched
    # fallback synthetic for now
    return synthetic_political()


if __name__ == "__main__":
    df = synthetic_political()
    df.to_csv(DATA_DIR / "political_risk_tower.csv", index=False)
    print(df)
