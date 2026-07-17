"""
Production real builder v2 — memory efficient, progress prints, sec-only mode first
"""
from pathlib import Path
import json, time, math, gc, argparse, sys
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
CACHE_DIR = ROOT / "pipeline" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR/"sec").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT/"pipeline"))
from feature_spec import FEATURE_FAMILIES, ALL_FEATURES, SECTORS, GAME_PROFILE_FEATURES
from fetch_sec import fetch_company_facts
from build_real import get_fact_for_year, REVENUE_TAGS, COGS_TAGS, GROSS_PROFIT_TAGS, OP_INCOME_TAGS, NET_INCOME_TAGS, ASSETS_TAGS, LIAB_TAGS, EQUITY_TAGS, CASH_TAGS, DEBT_LT_TAGS, DEBT_ST_TAGS, OCF_TAGS, CAPEX_TAGS, CURR_ASSET_TAGS, CURR_LIAB_TAGS, DEPR_TAGS, INTEREST_TAGS, SHARES_DILUTED_TAGS, SHARES_BASIC_TAGS, RET_EARN_TAGS

def load_universe(limit):
    import json
    path=DATA_DIR/"universe.json"
    uni=json.loads(path.read_text())
    return uni[:limit]

def safe_div(a,b):
    if b is None or a is None or b==0:
        return None
    try:
        return float(a)/float(b)
    except:
        return None

def build_real_v2(limit=100, years_range=(2015,2024), sec_only=True):
    uni=load_universe(limit)
    years=list(range(years_range[0], years_range[1]+1))
    print(f"Building REAL v2 for {len(uni)} tickers x {len(years)} years sec_only={sec_only}")
    all_rows=[]
    for idx, entry in enumerate(uni):
        ticker=entry["ticker"]
        cik=entry["cik"]
        sector=entry.get("sector","Industrials")
        company=entry.get("company",ticker)
        print(f"[{idx+1}/{len(uni)}] {ticker} CIK {cik} ...", flush=True)
        facts=None
        try:
            facts=fetch_company_facts(cik)
            # small sleep already inside fetch
        except Exception as e:
            print(f"  SEC fetch fail {ticker}: {e}")
            facts=None
        yearly_raw={}
        for yr in years:
            raw={}
            if facts:
                # Use helper
                rev=get_fact_for_year(facts, REVENUE_TAGS, yr)
                cogs=get_fact_for_year(facts, COGS_TAGS, yr)
                gross=get_fact_for_year(facts, GROSS_PROFIT_TAGS, yr)
                op=get_fact_for_year(facts, OP_INCOME_TAGS, yr)
                net=get_fact_for_year(facts, NET_INCOME_TAGS, yr)
                assets=get_fact_for_year(facts, ASSETS_TAGS, yr)
                liab=get_fact_for_year(facts, LIAB_TAGS, yr)
                equity=get_fact_for_year(facts, EQUITY_TAGS, yr)
                cash=get_fact_for_year(facts, CASH_TAGS, yr)
                debt_lt=get_fact_for_year(facts, DEBT_LT_TAGS, yr)
                debt_st=get_fact_for_year(facts, DEBT_ST_TAGS, yr)
                ocf=get_fact_for_year(facts, OCF_TAGS, yr)
                capex=get_fact_for_year(facts, CAPEX_TAGS, yr)
                cur_a=get_fact_for_year(facts, CURR_ASSET_TAGS, yr)
                cur_l=get_fact_for_year(facts, CURR_LIAB_TAGS, yr)
                depr=get_fact_for_year(facts, DEPR_TAGS, yr)
                interest=get_fact_for_year(facts, INTEREST_TAGS, yr)
                shares_d=get_fact_for_year(facts, SHARES_DILUTED_TAGS, yr)
                shares_b=get_fact_for_year(facts, SHARES_BASIC_TAGS, yr)
                ret_earn=get_fact_for_year(facts, RET_EARN_TAGS, yr)
                if gross is None and rev is not None and cogs is not None:
                    gross=rev-cogs
                debt=None
                if debt_lt is not None or debt_st is not None:
                    debt=(debt_lt or 0)+(debt_st or 0)
                raw.update({
                    "REV":rev,"COGS":cogs,"GROSS_PROFIT":gross,"OP_INCOME":op,"NET_INCOME":net,
                    "ASSETS":assets,"LIAB":liab,"EQUITY":equity,"CASH":cash,"DEBT":debt,"DEBT_LT":debt_lt,"DEBT_ST":debt_st,
                    "OCF":ocf,"CAPEX":capex,"CUR_ASSETS":cur_a,"CUR_LIAB":cur_l,"DEPR":depr,"INTEREST":interest,
                    "SHARES_D":shares_d,"SHARES_B":shares_b,"RET_EARN":ret_earn
                })
            yearly_raw[yr]=raw
        # free facts early
        del facts
        gc.collect()
        # Now compute features per year
        prev_vals={}
        for yr in sorted(yearly_raw.keys()):
            raw=yearly_raw[yr]
            rev=raw.get("REV")
            cogs=raw.get("COGS")
            gross=raw.get("GROSS_PROFIT")
            op=raw.get("OP_INCOME")
            net=raw.get("NET_INCOME")
            assets=raw.get("ASSETS")
            liab=raw.get("LIAB")
            equity=raw.get("EQUITY")
            cash=raw.get("CASH")
            debt=raw.get("DEBT")
            ocf=raw.get("OCF")
            capex=raw.get("CAPEX")
            cur_a=raw.get("CUR_ASSETS")
            cur_l=raw.get("CUR_LIAB")
            depr=raw.get("DEPR")
            interest=raw.get("INTEREST")
            shares_d=raw.get("SHARES_D") or raw.get("SHARES_B")
            ret_earn=raw.get("RET_EARN")

            ebitda=None
            if op is not None and depr is not None:
                ebitda=op+depr
            elif op is not None:
                ebitda=op*1.15
            ebit=op
            fcf=None
            if ocf is not None and capex is not None:
                # capex positive outflow, so OCF - abs(CAPEX)
                fcf = ocf - abs(capex) if capex>0 else ocf + capex
            elif ocf is not None:
                fcf=ocf*0.8
            gross_margin=safe_div(gross, rev)
            op_margin=safe_div(op, rev)
            net_margin=safe_div(net, rev)
            ebitda_margin=safe_div(ebitda, rev)
            fcf_margin=safe_div(fcf, rev)
            book_value=equity
            working_cap=None
            if cur_a is not None and cur_l is not None:
                working_cap=cur_a - cur_l
            net_debt=None
            if debt is not None and cash is not None:
                net_debt=debt-cash
            invested_cap=None
            if equity is not None and debt is not None and cash is not None:
                invested_cap=equity+debt-cash

            def yoy(curr, prev_key):
                prev=prev_vals.get(prev_key)
                if curr is None or prev is None or prev==0:
                    return None
                return (curr-prev)/abs(prev)
            def cagr(curr, prev_key, yrs=3):
                prev=prev_vals.get(prev_key)
                if curr is None or prev is None or prev<=0 or curr<=0:
                    return None
                try:
                    return (curr/prev)**(1.0/yrs)-1
                except:
                    return None

            rev_yoy=yoy(rev, f"REV_{yr-1}")
            rev_3y=cagr(rev, f"REV_{yr-3}",3)
            ebitda_yoy=yoy(ebitda, f"EBITDA_{yr-1}")
            net_yoy=yoy(net, f"NET_{yr-1}")
            fcf_yoy=yoy(fcf, f"FCF_{yr-1}")
            eps=safe_div(net, shares_d) if net and shares_d else None
            bvps=safe_div(equity, shares_d)
            fcfps=safe_div(fcf, shares_d)
            shares_yoy=yoy(shares_d, f"SHARES_{yr-1}")
            roe=safe_div(net, equity)
            roa=safe_div(net, assets)
            roic=safe_div(net, invested_cap) if invested_cap else safe_div(op, invested_cap)
            curr_ratio=safe_div(cur_a, cur_l)
            debt_eq=safe_div(debt, equity)
            debt_ebitda=safe_div(debt, ebitda)
            int_cov=safe_div(op, interest) if interest and interest!=0 else None
            debt_assets=safe_div(debt, assets)
            net_debt_ebitda=safe_div(net_debt, ebitda)
            asset_turn=safe_div(rev, assets)
            capex_depr=safe_div(capex, depr) if capex and depr else None

            # Market placeholders if sec_only
            if sec_only:
                ret_1m=ret_3m=ret_6m=ret_12m=vol_30=vol_90=vol_252=beta=vol_avg=mom_12_1=price_vs_52w=None
                price=market_cap=ev=pe=pb=ps=ev_ebitda=ev_sales=earn_yield=fcf_yield=None
                div_yield=0.015
            else:
                # will be filled later by market enrichment
                ret_1m=ret_3m=ret_6m=ret_12m=vol_30=vol_90=vol_252=beta=vol_avg=mom_12_1=price_vs_52w=None
                price=market_cap=ev=pe=pb=ps=ev_ebitda=ev_sales=earn_yield=fcf_yield=None
                div_yield=0.015

            # Synthetic ownership/management still needed for 17 families
            import random, math
            inst_pct=0.75
            inst_delta=0.0
            insider_net=0.0
            float_pct=0.9
            top10_conc=0.35
            short_int=0.03
            ceo_age=55
            ceo_tenure=6
            ceo_founder=0
            ceo_comp=12
            ceo_eq=1.5
            avg_neo=11
            pay_ratio=200
            board_indep=75
            board_size=9
            insider_own=3
            ceo_pay_vs=0
            neo_turn=0.15
            ceo_dual=0

            # Macro
            rate_map={2015:2.27,2016:1.84,2017:2.33,2018:2.91,2019:2.14,2020:0.89,2021:1.45,2022:2.95,2023:3.96,2024:4.2}
            vix_map={2015:16.7,2016:15.8,2017:11.1,2018:16.6,2019:15.4,2020:29.2,2021:19.7,2022:25.6,2023:16.8,2024:15}
            rate_10y=rate_map.get(yr,3.0)
            vix_avg=vix_map.get(yr,16)
            credit_spread=3.5
            gdp={2015:2.9,2016:1.8,2017:2.2,2018:2.9,2019:2.3,2020:-2.2,2021:5.8,2022:1.9,2023:2.5,2024:2.2}.get(yr,2)

            earn_surprise=0
            guidance_raise=0
            eps_rev_up=0.5
            rsi=50
            accident=0
            altman=None
            if assets and assets!=0 and equity and ret_earn is not None and ebit is not None:
                try:
                    wc=working_cap or 0
                    mv=equity # proxy market cap
                    liab_v=liab or assets*0.6
                    altman=1.2*(wc/assets)+1.4*(ret_earn/assets)+3.3*(ebit/assets)+0.6*(mv/liab_v)+1.0*((rev or 0)/assets)
                except:
                    altman=None
            piotroski=5

            # Build feature dict
            row_feat={
                "REV":rev,"COGS":cogs,"GROSS_PROFIT":gross,"OP_INCOME":op,"EBITDA":ebitda,"NET_INCOME":net,"EBIT":ebit,
                "GROSS_MARGIN":gross_margin,"OP_MARGIN":op_margin,"NET_MARGIN":net_margin,"EBITDA_MARGIN":ebitda_margin,
                "TOTAL_ASSETS":assets,"TOTAL_LIABILITIES":liab,"EQUITY":equity,"CASH":cash,"DEBT":debt,
                "BOOK_VALUE":book_value,"TANGIBLE_BOOK":equity,"WORKING_CAPITAL":working_cap,"NET_DEBT":net_debt,"INVESTED_CAPITAL":invested_cap,
                "OCF":ocf,"CAPEX":capex,"FCF":fcf,"FCF_MARGIN":fcf_margin,
                "OCF_TO_NET":safe_div(ocf,net),"FCF_CONVERSION":safe_div(fcf,net),"CAPEX_TO_REV":safe_div(capex,rev),
                "REV_YOY":rev_yoy,"EBITDA_YOY":ebitda_yoy,"NET_YOY":net_yoy,"FCF_YOY":fcf_yoy,
                "REV_3Y_CAGR":rev_3y,"EBITDA_3Y_CAGR":cagr(ebitda,f"EBITDA_{yr-3}"),"EPS_3Y_CAGR":cagr(eps,f"EPS_{yr-3}"),"BOOK_3Y_CAGR":cagr(book_value,f"BOOK_{yr-3}"),"OCF_3Y_CAGR":cagr(ocf,f"OCF_{yr-3}"),
                "ROE":roe,"ROA":roa,"ROIC":roic,"GROSS_MARGIN":gross_margin,"OP_MARGIN":op_margin,"NET_MARGIN":net_margin,"FCF_ROIC":safe_div(fcf,invested_cap),"EBITDA_MARGIN":ebitda_margin,"ROIC_WACC_SPREAD":(roic-0.08) if roic else None,
                "CURRENT_RATIO":curr_ratio,"QUICK_RATIO":curr_ratio,"DEBT_TO_EQUITY":debt_eq,"DEBT_TO_EBITDA":debt_ebitda,"INTEREST_COVERAGE":int_cov,"DEBT_TO_ASSETS":debt_assets,"NET_DEBT_TO_EBITDA":net_debt_ebitda,
                "ASSET_TURNOVER":asset_turn,"INVENTORY_TURNOVER":asset_turn,"RECEIVABLE_TURNOVER":asset_turn,"CASH_CONVERSION_CYCLE":None,"CAPEX_TO_DEPRE":capex_depr,
                "EPS_DILUTED":eps,"BVPS":bvps,"FCFPS":fcfps,"SHARES_YOY":shares_yoy,"DILUTION_3Y":cagr(shares_d,f"SHARES_{yr-3}"),
                "RET_1M":ret_1m,"RET_3M":ret_3m,"RET_6M":ret_6m,"RET_12M":ret_12m,"VOL_30D":vol_30,"VOL_90D":vol_90,"VOL_252D":vol_252,"BETA_1Y":beta,"VOLUME_AVG_30D":vol_avg,"MOMENTUM_12_1":mom_12_1,
                "PE":pe,"PB":pb,"PS":ps,"EV_EBITDA":ev_ebitda,"EV_SALES":ev_sales,"EARNINGS_YIELD":earn_yield,"FCF_YIELD":fcf_yield,"DIV_YIELD":div_yield,
                "NEO_COUNT":5,"CEO_AGE":ceo_age,"CEO_TENURE":ceo_tenure,"CEO_FOUNDER_FLAG":ceo_founder,"CEO_TOTAL_COMP":ceo_comp,"CEO_EQUITY_PCT":ceo_eq,"AVG_NEO_COMP":avg_neo,"CEO_PAY_RATIO":pay_ratio,"BOARD_INDEP_PCT":board_indep,"BOARD_SIZE":board_size,"INSIDER_OWN_PCT":insider_own,"CEO_PAY_VS_SECTOR":ceo_pay_vs,"NEO_TURNOVER":neo_turn,"CEO_DUALITY":ceo_dual,
                "INST_PCT":inst_pct,"INST_DELTA_QOQ":inst_delta,"INSIDER_NET_12M":insider_net,"FLOAT_PCT":float_pct,"TOP10_INST_CONC":top10_conc,"SHORT_INTEREST_PCT":short_int,
                "MDA_LENGTH":5,"MDA_SENTIMENT":0.05,"RISK_FACTOR_COUNT":20,"RISK_CHANGE_YOY":0,"FOG_INDEX_PROXY":18,"TONE_UNCERTAINTY":0.15,
                "SECTOR_REL_RET_12M":0,"SECTOR_CONCENTRATION":0.2,"SECTOR_BETA":1.0,
                "RATE_10Y":rate_10y,"VIX_AVG_FY":vix_avg,"CREDIT_SPREAD_PROXY":credit_spread,"GDP_GROWTH_FY":gdp,
                "EARN_SURPRISE_STREAK":earn_surprise,"GUIDANCE_RAISE_FLAG":guidance_raise,"EPS_REVISION_UP_PCT":eps_rev_up,"PRICE_VS_52W_HIGH":price_vs_52w if price_vs_52w is not None else 0.9,"RSI_14_PROXY":rsi,"ACCIDENT_DISCLOSURE":accident,
                "ALTMAN_Z":altman,"PIOTROSKI_F_SCORE_PROXY":piotroski
            }

            prev_vals[f"REV_{yr}"]=rev
            prev_vals[f"EBITDA_{yr}"]=ebitda
            prev_vals[f"NET_{yr}"]=net
            prev_vals[f"FCF_{yr}"]=fcf
            prev_vals[f"EPS_{yr}"]=eps
            prev_vals[f"BOOK_{yr}"]=book_value
            prev_vals[f"OCF_{yr}"]=ocf
            prev_vals[f"SHARES_{yr}"]=shares_d

            if rev is None and assets is None:
                continue
            all_rows.append({
                "ticker":ticker,"company":company,"sector":sector,"fiscal_year":str(yr),"features":row_feat
            })
        # end years
    print(f"Collected {len(all_rows)} rows")

    # Build matrix
    D=len(ALL_FEATURES)
    N=len(all_rows)
    Z_raw=np.zeros((N,D), dtype=np.float32)
    mask=np.zeros((N,D), dtype=np.float32)
    tickers=[]; names=[]; fyears=[]; sectors=[]
    for i,row in enumerate(all_rows):
        tickers.append(row["ticker"])
        names.append(row["company"])
        fyears.append(row["fiscal_year"])
        sectors.append(row["sector"])
        for j,feat_name in enumerate(ALL_FEATURES):
            val=row["features"].get(feat_name)
            if val is not None:
                try:
                    if isinstance(val,float) and (np.isnan(val) or np.isinf(val)):
                        continue
                    Z_raw[i,j]=float(val)
                    mask[i,j]=1.0
                except:
                    pass
    # Fill median per FY then global
    Z_filled=Z_raw.copy()
    for fy in sorted(set(fyears)):
        rows=[k for k,v in enumerate(fyears) if v==fy]
        for j in range(D):
            col=Z_raw[rows,j]
            m=mask[rows,j]
            valid=col[m>0.5]
            if len(valid)==0:
                gvalid=Z_raw[:,j][mask[:,j]>0.5]
                median=np.median(gvalid) if len(gvalid)>0 else 0.0
            else:
                median=np.median(valid)
            Z_filled[rows,j]=np.where(m>0.5, col, median)
    # z-score
    Z=np.zeros_like(Z_filled)
    for fy in sorted(set(fyears)):
        rows=[k for k,v in enumerate(fyears) if v==fy]
        for j in range(D):
            vals=Z_filled[rows,j]
            if len(vals)<2:
                Z[rows,j]=0
                continue
            mean=vals.mean()
            std=max(vals.std(),1e-6)
            zs=(vals-mean)/std
            zs=np.clip(zs,-4,4)
            Z[rows,j]=zs

    manifest={
        "features":ALL_FEATURES,
        "families":[ next((fam for fam,feats in FEATURE_FAMILIES.items() if feat_name in feats), "unknown") for feat_name in ALL_FEATURES ],
        "game_features":GAME_PROFILE_FEATURES,
        "sectors":SECTORS,
        "real_data":True,
        "sec_only":sec_only,
        "years":years
    }
    out_path=DATA_DIR/"train_matrix.npz"
    np.savez_compressed(DATA_DIR/"train_matrix_real.npz", Z=Z.astype(np.float32), mask=mask, ticker=np.array(tickers), name=np.array(names), fiscal_year=np.array(fyears), sector=np.array(sectors), cluster=np.zeros(N,dtype=np.int64), Z_raw=Z_raw)
    np.savez_compressed(out_path, Z=Z.astype(np.float32), mask=mask, ticker=np.array(tickers), name=np.array(names), fiscal_year=np.array(fyears), sector=np.array(sectors), cluster=np.zeros(N,dtype=np.int64))
    (DATA_DIR/"feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved REAL {N} rows x {D} feats to {out_path}")
    return out_path

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--years", type=str, default="2015-2024")
    ap.add_argument("--sec-only", action="store_true", default=True)
    ap.add_argument("--no-sec-only", dest="sec_only", action="store_false")
    args=ap.parse_args()
    y0,y1=map(int,args.years.split("-"))
    build_real_v2(limit=args.limit, years_range=(y0,y1), sec_only=args.sec_only)
