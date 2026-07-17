"""Fetch SEC submissions.json for each CIK — contains DEF14A and Form4 index"""
import json, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"pipeline"))
from fetch_sec import robust_fetch_json, USER_AGENT

CACHE=ROOT/"pipeline"/"cache"/"sec_submissions"
CACHE.mkdir(parents=True, exist_ok=True)

def fetch_sub(cik_pad):
    out=CACHE/f"sub_{cik_pad}.json"
    if out.exists() and out.stat().st_size>5000:
        try:
            return json.loads(out.read_text())
        except:
            out.unlink(missing_ok=True)
    url=f"https://data.sec.gov/submissions/CIK{cik_pad}.json"
    print(f"Fetching submissions CIK {cik_pad} {url}")
    data=robust_fetch_json(url)
    if data:
        try:
            out.write_text(json.dumps(data))
        except Exception as e:
            print(f"write fail {e}")
        time.sleep(0.25)
    else:
        print(f"FAIL {cik_pad}")
        time.sleep(1)
    return data

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    args=ap.parse_args()
    uni=json.loads((ROOT/"pipeline"/"data"/"universe.json").read_text())
    subset=uni[args.start:args.start+args.limit]
    print(f"Top {len(subset)} from {args.start}")
    ok=0
    for entry in subset:
        cik=str(entry["cik"]).zfill(10)
        ticker=entry["ticker"]
        data=fetch_sub(cik)
        if data:
            ok+=1
            filings=data.get("filings",{}).get("recent",{})
            forms=filings.get("form",[])
            # count DEF14A and Form4
            def14a=[f for f in forms if "DEF 14A" in f or "DEF14A" in f]
            form4=[f for f in forms if f=="4"]
            print(f"  {ticker} {cik} DEF14A:{len(def14a)} Form4:{len(form4)} recent total {len(forms)}")
        else:
            print(f"  {ticker} FAIL")
    print(f"Done {ok}/{len(subset)}")
