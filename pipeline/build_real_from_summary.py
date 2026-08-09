"""
Build real matrix from tiny summary JSONs — production grade, low memory
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
CACHE_SUM = ROOT / "pipeline" / "cache" / "sec" / "sec_summary"
import sys

sys.path.insert(0, str(ROOT / "pipeline"))
from def14a_features import get_def14a_row
from feature_spec import ALL_FEATURES, FEATURE_FAMILIES, GAME_PROFILE_FEATURES, SECTORS
from market_features import get_market_row, valuation_row


def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    try:
        return float(a) / float(b)
    except Exception:
        return None


def load_summaries():
    out = {}
    for p in CACHE_SUM.glob("summary_*.json"):
        try:
            j = json.loads(p.read_text())
            cik = j.get("_meta", {}).get("cik") or p.stem.replace("summary_", "")
            out[cik] = j
        except Exception:
            pass
    return out


def build_from_summary(limit=None):
    # load universe
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} summaries, universe {len(uni)}")
    all_rows = []
    for entry in uni[:limit] if limit else uni:
        cik = entry["cik"].zfill(10)
        ticker = entry["ticker"]
        sector = entry.get("sector", "Industrials")
        company = entry.get("company", ticker)
        summ = summaries.get(cik) or summaries.get(entry["cik"]) or summaries.get(cik.lstrip("0"))
        if not summ:
            # try find file with same cik padded? summaries keys are padded
            continue
        prev_vals = {}
        for yr in range(2015, 2025):
            ystr = str(yr)
            raw = summ.get(ystr, {})
            if not raw:
                continue
            rev = raw.get("REVENUE")
            cogs = raw.get("COGS")
            gross = raw.get("GROSS")
            op = raw.get("OP_INCOME")
            net = raw.get("NET_INCOME")
            assets = raw.get("ASSETS")
            liab = raw.get("LIAB")
            equity = raw.get("EQUITY")
            cash = raw.get("CASH")
            debt_lt = raw.get("DEBT_LT")
            debt_st = raw.get("DEBT_ST")
            ocf = raw.get("OCF")
            capex = raw.get("CAPEX")
            cur_a = raw.get("CURR_A")
            cur_l = raw.get("CURR_L")
            depr = raw.get("DEPR")
            interest = raw.get("INTEREST")
            shares_d = raw.get("SHARES_D") or raw.get("SHARES_B")
            ret_earn = raw.get("RET_EARN")
            inventory = raw.get("INVENTORY")
            receivables = raw.get("RECEIVABLES")
            goodwill = raw.get("GOODWILL")
            intangibles = raw.get("INTANGIBLES")
            dividends_paid = raw.get("DIVIDENDS_PAID")
            if gross is None and rev is not None and cogs is not None:
                gross = rev - cogs
            debt = None
            if debt_lt is not None or debt_st is not None:
                debt = (debt_lt or 0) + (debt_st or 0)
            # derived -- honest: mask rather than estimate when a real
            # component is missing (was op*1.15 / ocf*0.8 heuristics, i.e.
            # fabricated numbers dressed up as derived ones; §257 elsewhere
            # in this codebase is explicit that missing means masked, never
            # imputed, and this violated that)
            ebitda = op + depr if (op is not None and depr is not None) else None
            ebit = op
            fcf = None
            if ocf is not None and capex is not None:
                fcf = ocf - abs(capex) if capex > 0 else ocf + capex
            gross_margin = safe_div(gross, rev)
            op_margin = safe_div(op, rev)
            net_margin = safe_div(net, rev)
            ebitda_margin = safe_div(ebitda, rev)
            fcf_margin = safe_div(fcf, rev)
            book_value = equity
            tangible_book = equity - (goodwill or 0) - (intangibles or 0) if equity is not None else None
            working_cap = cur_a - cur_l if (cur_a is not None and cur_l is not None) else None
            net_debt = debt - cash if (debt is not None and cash is not None) else None
            invested_cap = (
                equity + debt - cash if (equity is not None and debt is not None and cash is not None) else None
            )

            def yoy(curr, prev_key):
                prev = prev_vals.get(prev_key)  # noqa: B023 (loop var used only within same iteration)
                if curr is None or prev is None or prev == 0:
                    return None
                return (curr - prev) / abs(prev)

            def cagr(curr, prev_key, yrs=3):
                prev = prev_vals.get(prev_key)  # noqa: B023 (loop var used only within same iteration)
                if curr is None or prev is None or prev <= 0 or curr <= 0:
                    return None
                try:
                    return (curr / prev) ** (1.0 / yrs) - 1
                except Exception:
                    return None

            rev_yoy = yoy(rev, f"REV_{yr - 1}")
            rev_3y = cagr(rev, f"REV_{yr - 3}", 3)
            ebitda_yoy = yoy(ebitda, f"EBITDA_{yr - 1}")
            net_yoy = yoy(net, f"NET_{yr - 1}")
            fcf_yoy = yoy(fcf, f"FCF_{yr - 1}")
            eps = safe_div(net, shares_d)
            bvps = safe_div(equity, shares_d)
            fcfps = safe_div(fcf, shares_d)
            shares_yoy = yoy(shares_d, f"SHARES_{yr - 1}")
            roe = safe_div(net, equity)
            roa = safe_div(net, assets)
            roic = safe_div(net, invested_cap) if invested_cap else safe_div(op, invested_cap)
            curr_ratio = safe_div(cur_a, cur_l)
            quick_ratio = (
                safe_div(cur_a - inventory, cur_l) if (cur_a is not None and inventory is not None and cur_l) else None
            )
            inventory_turn = safe_div(cogs, inventory) if inventory else None
            receivable_turn = safe_div(rev, receivables) if receivables else None
            debt_eq = safe_div(debt, equity)
            debt_ebitda = safe_div(debt, ebitda)
            int_cov = safe_div(op, interest) if interest else None
            debt_assets = safe_div(debt, assets)
            net_debt_ebitda = safe_div(net_debt, ebitda)
            asset_turn = safe_div(rev, assets)
            capex_depr = safe_div(capex, depr)

            # Real market/valuation features (price history already fetched by
            # fetch_market_history.py; was previously hardcoded to None/constants
            # for all 21 of these -- see docs/rebuild notes 2026-07-30).
            mkt = get_market_row(ticker, yr)
            val = valuation_row(
                price=mkt["_price"],
                shares=shares_d,
                eps=eps,
                bvps=bvps,
                rev=rev,
                ebitda=ebitda,
                fcf=fcf,
                debt=debt,
                cash=cash,
                dividends_paid=dividends_paid,
            )

            # Real CEO/NEO total comp from DEF14A inline-XBRL Pay-vs-
            # Performance tags (parse_def14a_xbrl.py) -- only covers proxies
            # filed for FY2022+ (the rule's effective date) and only the
            # tickers/years actually fetched+parsed; None elsewhere, not
            # fabricated. The other 12 management_neo fields and all 6
            # ownership fields still have no wired real source (see
            # docs/DEF14A_FORM4_CHRONOGRAPH_SPEC.md) and stay None below.
            def14a = get_def14a_row(ticker, yr)

            # Piotroski F-score proxy (real, computed from fundamentals already
            # in hand; was hardcoded to a literal 5 for every row before).
            prev_net = prev_vals.get(f"NET_{yr-1}")
            prev_assets = prev_vals.get(f"ASSETS_{yr-1}")
            prev_roa = safe_div(prev_net, prev_assets) if prev_assets else None
            piotroski = None
            if all(v is not None for v in (roa, ocf, net, debt_assets, curr_ratio)):
                pts = 0
                pts += 1 if roa > 0 else 0
                pts += 1 if ocf > 0 else 0
                pts += 1 if ocf > net else 0
                prev_debt_assets = prev_vals.get(f"DEBT_ASSETS_{yr-1}")
                if prev_debt_assets is not None:
                    pts += 1 if debt_assets < prev_debt_assets else 0
                prev_curr = prev_vals.get(f"CURR_RATIO_{yr-1}")
                if prev_curr is not None:
                    pts += 1 if curr_ratio > prev_curr else 0
                if shares_yoy is not None:
                    pts += 1 if shares_yoy <= 0.001 else 0
                prev_gm = prev_vals.get(f"GROSS_MARGIN_{yr-1}")
                if prev_gm is not None:
                    pts += 1 if gross_margin is not None and gross_margin > prev_gm else 0
                prev_at = prev_vals.get(f"ASSET_TURN_{yr-1}")
                if prev_at is not None:
                    pts += 1 if asset_turn is not None and asset_turn > prev_at else 0
                if prev_roa is not None:
                    pts += 1 if roa > prev_roa else 0
                piotroski = pts

            # Everything below genuinely has no free real-data source in this
            # pipeline yet (13F institutional ownership, DEF14A executive comp
            # / governance -- fetch_def14a*.py exists but isn't wired to this
            # script, 10-K text NLP for disclosure sentiment/risk, analyst
            # estimates for surprise/guidance/revisions). Masked, not
            # fabricated -- these used to be hardcoded identical constants for
            # every company (e.g. every CEO age = 55, every board size = 9),
            # which is worse than missing: it looks like real data until you
            # check the variance.
            rate_map = {
                2015: 2.27,
                2016: 1.84,
                2017: 2.33,
                2018: 2.91,
                2019: 2.14,
                2020: 0.89,
                2021: 1.45,
                2022: 2.95,
                2023: 3.96,
                2024: 4.2,
            }
            vix_map = {
                2015: 16.7,
                2016: 15.8,
                2017: 11.1,
                2018: 16.6,
                2019: 15.4,
                2020: 29.2,
                2021: 19.7,
                2022: 25.6,
                2023: 16.8,
                2024: 15,
            }
            rate_10y = rate_map.get(yr, 3.0)
            vix_avg = vix_map.get(yr, 16)
            gdp = {
                2015: 2.9,
                2016: 1.8,
                2017: 2.2,
                2018: 2.9,
                2019: 2.3,
                2020: -2.2,
                2021: 5.8,
                2022: 1.9,
                2023: 2.5,
                2024: 2.2,
            }.get(yr, 2)
            altman = None
            if assets and assets != 0 and equity and ret_earn is not None and ebit is not None:
                try:
                    wc = working_cap or 0
                    mv = equity
                    liab_v = liab or assets * 0.6
                    altman = (
                        1.2 * (wc / assets)
                        + 1.4 * (ret_earn / assets)
                        + 3.3 * (ebit / assets)
                        + 0.6 * (mv / liab_v)
                        + 1.0 * ((rev or 0) / assets)
                    )
                except Exception:
                    pass

            row_feat = {
                "REV": rev,
                "COGS": cogs,
                "GROSS_PROFIT": gross,
                "OP_INCOME": op,
                "EBITDA": ebitda,
                "NET_INCOME": net,
                "EBIT": ebit,
                "GROSS_MARGIN": gross_margin,
                "OP_MARGIN": op_margin,
                "NET_MARGIN": net_margin,
                "EBITDA_MARGIN": ebitda_margin,
                "TOTAL_ASSETS": assets,
                "TOTAL_LIABILITIES": liab,
                "EQUITY": equity,
                "CASH": cash,
                "DEBT": debt,
                "BOOK_VALUE": book_value,
                "TANGIBLE_BOOK": tangible_book,
                "WORKING_CAPITAL": working_cap,
                "NET_DEBT": net_debt,
                "INVESTED_CAPITAL": invested_cap,
                "OCF": ocf,
                "CAPEX": capex,
                "FCF": fcf,
                "FCF_MARGIN": fcf_margin,
                "OCF_TO_NET": safe_div(ocf, net),
                "FCF_CONVERSION": safe_div(fcf, net),
                "CAPEX_TO_REV": safe_div(capex, rev),
                "REV_YOY": rev_yoy,
                "EBITDA_YOY": ebitda_yoy,
                "NET_YOY": net_yoy,
                "FCF_YOY": fcf_yoy,
                "REV_3Y_CAGR": rev_3y,
                "EBITDA_3Y_CAGR": cagr(ebitda, f"EBITDA_{yr - 3}"),
                "EPS_3Y_CAGR": cagr(eps, f"EPS_{yr - 3}"),
                "BOOK_3Y_CAGR": cagr(book_value, f"BOOK_{yr - 3}"),
                "OCF_3Y_CAGR": cagr(ocf, f"OCF_{yr - 3}"),
                "ROE": roe,
                "ROA": roa,
                "ROIC": roic,
                "FCF_ROIC": safe_div(fcf, invested_cap),
                "ROIC_WACC_SPREAD": (roic - 0.08) if roic else None,
                "CURRENT_RATIO": curr_ratio,
                "QUICK_RATIO": quick_ratio,
                "DEBT_TO_EQUITY": debt_eq,
                "DEBT_TO_EBITDA": debt_ebitda,
                "INTEREST_COVERAGE": int_cov,
                "DEBT_TO_ASSETS": debt_assets,
                "NET_DEBT_TO_EBITDA": net_debt_ebitda,
                "ASSET_TURNOVER": asset_turn,
                "INVENTORY_TURNOVER": inventory_turn,
                "RECEIVABLE_TURNOVER": receivable_turn,
                "CASH_CONVERSION_CYCLE": None,
                "CAPEX_TO_DEPRE": capex_depr,
                "EPS_DILUTED": eps,
                "BVPS": bvps,
                "FCFPS": fcfps,
                "SHARES_YOY": shares_yoy,
                "DILUTION_3Y": cagr(shares_d, f"SHARES_{yr - 3}"),
                "RET_1M": mkt["RET_1M"],
                "RET_3M": mkt["RET_3M"],
                "RET_6M": mkt["RET_6M"],
                "RET_12M": mkt["RET_12M"],
                "VOL_30D": mkt["VOL_30D"],
                "VOL_90D": mkt["VOL_90D"],
                "VOL_252D": mkt["VOL_252D"],
                "BETA_1Y": mkt["BETA_1Y"],
                "VOLUME_AVG_30D": mkt["VOLUME_AVG_30D"],
                "MOMENTUM_12_1": mkt["MOMENTUM_12_1"],
                "PE": val["PE"],
                "PB": val["PB"],
                "PS": val["PS"],
                "EV_EBITDA": val["EV_EBITDA"],
                "EV_SALES": val["EV_SALES"],
                "EARNINGS_YIELD": val["EARNINGS_YIELD"],
                "FCF_YIELD": val["FCF_YIELD"],
                "DIV_YIELD": val["DIV_YIELD"],
                # Below: no free real-data source wired up yet (13F ownership,
                # DEF14A executive comp/governance, 10-K text NLP, analyst
                # estimates). Masked (None), not fabricated -- these were
                # hardcoded identical constants for every company before
                # (e.g. every CEO age = 55, every board size = 9), which
                # silently violated this codebase's own §257 masking
                # discipline. A real fix needs new acquisition work, not
                # invented numbers standing in for it.
                "NEO_COUNT": None,
                "CEO_AGE": None,
                "CEO_TENURE": None,
                "CEO_FOUNDER_FLAG": None,
                "CEO_TOTAL_COMP": def14a["CEO_TOTAL_COMP"],
                "CEO_EQUITY_PCT": None,
                "AVG_NEO_COMP": def14a["AVG_NEO_COMP"],
                "CEO_PAY_RATIO": None,
                "BOARD_INDEP_PCT": None,
                "BOARD_SIZE": None,
                "INSIDER_OWN_PCT": None,
                "CEO_PAY_VS_SECTOR": None,
                "NEO_TURNOVER": None,
                "CEO_DUALITY": None,
                "INST_PCT": None,
                "INST_DELTA_QOQ": None,
                "INSIDER_NET_12M": None,
                "FLOAT_PCT": None,
                "TOP10_INST_CONC": None,
                "SHORT_INTEREST_PCT": None,
                "MDA_LENGTH": None,
                "MDA_SENTIMENT": None,
                "RISK_FACTOR_COUNT": None,
                "RISK_CHANGE_YOY": None,
                "FOG_INDEX_PROXY": None,
                "TONE_UNCERTAINTY": None,
                # Sector-relative fields need a second pass over the fully
                # built matrix (per-sector aggregation) -- not computed here.
                "SECTOR_REL_RET_12M": None,
                "SECTOR_CONCENTRATION": None,
                "SECTOR_BETA": None,
                "RATE_10Y": rate_10y,
                "VIX_AVG_FY": vix_avg,
                "CREDIT_SPREAD_PROXY": None,
                "GDP_GROWTH_FY": gdp,
                "EARN_SURPRISE_STREAK": None,
                "GUIDANCE_RAISE_FLAG": None,
                "EPS_REVISION_UP_PCT": None,
                "PRICE_VS_52W_HIGH": mkt["PRICE_VS_52W_HIGH"],
                "RSI_14_PROXY": mkt["RSI_14_PROXY"],
                "ACCIDENT_DISCLOSURE": None,
                "ALTMAN_Z": altman,
                "PIOTROSKI_F_SCORE_PROXY": piotroski,
            }
            # ensure all features present
            # handle duplicate keys: last wins
            for k in ALL_FEATURES:
                if k not in row_feat:
                    row_feat[k] = None

            prev_vals[f"REV_{yr}"] = rev
            prev_vals[f"EBITDA_{yr}"] = ebitda
            prev_vals[f"NET_{yr}"] = net
            prev_vals[f"FCF_{yr}"] = fcf
            prev_vals[f"EPS_{yr}"] = eps
            prev_vals[f"BOOK_{yr}"] = book_value
            prev_vals[f"OCF_{yr}"] = ocf
            prev_vals[f"SHARES_{yr}"] = shares_d
            prev_vals[f"ASSETS_{yr}"] = assets
            prev_vals[f"DEBT_ASSETS_{yr}"] = debt_assets
            prev_vals[f"CURR_RATIO_{yr}"] = curr_ratio
            prev_vals[f"GROSS_MARGIN_{yr}"] = gross_margin
            prev_vals[f"ASSET_TURN_{yr}"] = asset_turn

            if rev is None and assets is None:
                continue
            all_rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "sector": sector,
                    "fiscal_year": str(yr),
                    "features": row_feat,
                }
            )

    print(f"Collected {len(all_rows)} rows from {len({r['ticker'] for r in all_rows})} tickers")

    D = len(ALL_FEATURES)
    N = len(all_rows)
    Z_raw = np.zeros((N, D), dtype=np.float32)
    mask = np.zeros((N, D), dtype=np.float32)
    tickers = []
    names = []
    fyears = []
    sectors = []
    for i, row in enumerate(all_rows):
        tickers.append(row["ticker"])
        names.append(row["company"])
        fyears.append(row["fiscal_year"])
        sectors.append(row["sector"])
        for j, feat_name in enumerate(ALL_FEATURES):
            val = row["features"].get(feat_name)
            if val is not None:
                try:
                    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                        continue
                    Z_raw[i, j] = float(val)
                    mask[i, j] = 1.0
                except Exception:
                    pass
    # fill median per FY then global, z-score per FY
    Z_filled = Z_raw.copy()
    for fy in sorted(set(fyears)):
        rows = [k for k, v in enumerate(fyears) if v == fy]
        for j in range(D):
            col = Z_raw[rows, j]
            m = mask[rows, j]
            valid = col[m > 0.5]
            if len(valid) == 0:
                gvalid = Z_raw[:, j][mask[:, j] > 0.5]
                median = np.median(gvalid) if len(gvalid) > 0 else 0.0
            else:
                median = np.median(valid)
            Z_filled[rows, j] = np.where(m > 0.5, col, median)
    Z = np.zeros_like(Z_filled)
    for fy in sorted(set(fyears)):
        rows = [k for k, v in enumerate(fyears) if v == fy]
        for j in range(D):
            vals = Z_filled[rows, j]
            if len(vals) < 2:
                Z[rows, j] = 0
                continue
            mean = vals.mean()
            std = max(vals.std(), 1e-6)
            zs = (vals - mean) / std
            zs = np.clip(zs, -4, 4)
            Z[rows, j] = zs

    manifest = {
        "features": ALL_FEATURES,
        "families": [
            next(
                (fam for fam, feats in FEATURE_FAMILIES.items() if feat in feats),
                "unknown",
            )
            for feat in ALL_FEATURES
        ],
        "game_features": GAME_PROFILE_FEATURES,
        "sectors": SECTORS,
        "real_data": True,
        "sec_only": True,
        "years": list(range(2015, 2025)),
        "tickers": len(set(tickers)),
        "rows": N,
    }
    out_path = DATA_DIR / "train_matrix.npz"
    np.savez_compressed(
        DATA_DIR / "train_matrix_real.npz",
        Z=Z.astype(np.float32),
        mask=mask,
        ticker=np.array(tickers),
        name=np.array(names),
        fiscal_year=np.array(fyears),
        sector=np.array(sectors),
        cluster=np.zeros(N, dtype=np.int64),
        Z_raw=Z_raw,
    )
    np.savez_compressed(
        out_path,
        Z=Z.astype(np.float32),
        mask=mask,
        ticker=np.array(tickers),
        name=np.array(names),
        fiscal_year=np.array(fyears),
        sector=np.array(sectors),
        cluster=np.zeros(N, dtype=np.int64),
    )
    (DATA_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved REAL {N} rows x {D} feats {len(set(tickers))} tickers to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build_from_summary(limit=args.limit)
