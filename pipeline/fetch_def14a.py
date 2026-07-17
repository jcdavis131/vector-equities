"""Fetch DEF14A HTML for top N tickers based on submissions.json"""
import json, time, sys, re, os, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"pipeline"))
from fetch_sec import USER_AGENT, CACHE
# Reuse curl logic from fetch_sec?
import urllib.request

CACHE_SUB=ROOT/"pipeline"/"cache"/"sec_submissions"
CACHE_DEF=ROOT/"pipeline"/"cache"/"sec_def14a"
CACHE_DEF.mkdir(parents=True, exist_ok=True)
(CACHE_DEF/"parsed").mkdir(parents=True, exist_ok=True)

def fetch_url_to_file(url, out_path, max_retries=3):
    if out_path.exists() and out_path.stat().st_size>5000:
        return True
    for attempt in range(max_retries):
        try:
            req=urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept":"text/html,application/xhtml+xml"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data=r.read()
                out_path.write_bytes(data)
                print(f"  fetched {url} -> {out_path} {len(data)}")
                time.sleep(0.3)
                return True
        except Exception as e:
            # curl fallback
            try:
                tmp="/tmp/def14a_tmp.html"
                cmd=["curl","-sL","--max-time","90","-H",f"User-Agent: {USER_AGENT}","-H","Accept: text/html","-o",tmp,url]
                import subprocess
                subprocess.run(cmd, timeout=95, capture_output=True)
                if os.path.exists(tmp) and os.path.getsize(tmp)>5000:
                    out_path.write_bytes(open(tmp,'rb').read())
                    print(f"  curl fetched {url} {out_path.stat().st_size}")
                    time.sleep(0.3)
                    return True
            except Exception as ce:
                pass
            print(f"  retry {attempt} fail {url} {e}")
            time.sleep(1+attempt)
    return False

def process_ticker(entry):
    cik_pad=str(entry["cik"]).zfill(10)
    cik_nopad=str(int(entry["cik"]))  # no leading zeros for archives path
    ticker=entry["ticker"]
    sub_path=CACHE_SUB/f"sub_{cik_pad}.json"
    if not sub_path.exists():
        print(f"No submissions for {ticker} {cik_pad}")
        return 0
    data=json.loads(sub_path.read_text())
    filings=data.get("filings",{}).get("recent",{})
    forms=filings.get("form",[])
    accession=filings.get("accessionNumber",[])
    filingDates=filings.get("filingDate",[])
    primaryDocs=filings.get("primaryDocument",[])
    count=0
    for i, form in enumerate(forms):
        if "DEF 14A" not in form and "DEF14A" not in form:
            continue
        # filter date 2015-01-01 to 2024-12-31 (include 2024 filing for FY2023)
        fdate=filingDates[i] if i < len(filingDates) else ""
        if fdate:
            try:
                year=int(fdate[:4])
                if year<2015 or year>2025:
                    continue
            except:
                pass
        acc=accession[i]
        acc_nodash=acc.replace("-","")
        primary=primaryDocs[i] if i < len(primaryDocs) else ""
        # build URL
        url=f"https://www.sec.gov/Archives/edgar/data/{cik_nopad}/{acc_nodash}/{primary}"
        # out file
        fy_year=str(year-1) if year else fdate  # filing year -1 = FY (approx)
        # for simplicity use filing year as FY label, will refine later using text
        out=CACHE_DEF/f"{ticker}_{fdate}_{acc}.html"
        # also link by FY later
        if fetch_url_to_file(url, out):
            count+=1
        # limit per ticker to 12 to avoid overload
        if count>=12:
            break
    print(f"{ticker} DEF14A fetched {count}")
    return count

if __name__=="__main__":
    import argparse, json
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    args=ap.parse_args()
    uni=json.loads((ROOT/"pipeline"/"data"/"universe.json").read_text())
    subset=uni[args.start:args.start+args.limit]
    total=0
    for entry in subset:
        c=process_ticker(entry)
        total+=c
    print(f"Total DEF14A files fetched: {total}")
