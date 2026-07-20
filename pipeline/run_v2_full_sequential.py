#!/usr/bin/env python3
import json, time, gc, sys, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=ROOT/"pipeline"/"data"
CACHE=ROOT/"pipeline"/"cache"
sys.path.insert(0, str(ROOT/"pipeline"))
from ingest_sec_v2 import load_universe, process_ticker, CHUNKS_V2_DIR, DATA_DIR

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--universe", default="")
    parser.add_argument("--filing-types", default="10-K")
    parser.add_argument("--force", action="store_true")
    args=parser.parse_args()
    
    uni_file = pathlib.Path(args.universe) if args.universe else None
    universe = load_universe(uni_file) if not uni_file else load_universe(uni_file)
    # prioritize big caps already done skip?
    filing_limits={}
    for ft in [s.strip().upper() for s in args.filing_types.split(",") if s.strip()]:
        if ft=="10-K":
            filing_limits["10-K"]=1
        elif ft in ["DEF14A","DEF 14A"]:
            filing_limits["DEF14A"]=1
            filing_limits["DEF 14A"]=1
        elif ft=="10-Q":
            filing_limits["10-Q"]=2
        elif ft=="8-K":
            filing_limits["8-K"]=5
        else:
            filing_limits[ft]=1
    if not filing_limits:
        filing_limits={"10-K":1}
    # slice
    subset = universe[args.start: args.start+args.limit] if args.limit!=0 else universe[args.start:]
    print(f"RUN sequential {len(subset)} from {args.start} limits={filing_limits}")
    ok=0
    fail=0
    total_chunks=0
    manifest_path = DATA_DIR / "sec_v2_manifest.json"
    existing={}
    if manifest_path.exists():
        try:
            m=json.loads(manifest_path.read_text())
            for e in m.get("entries",[]):
                existing[e["ticker"]]=e
        except:
            pass
    for idx, entry in enumerate(subset):
        ticker=entry["ticker"]
        # skip if already has chunks and not force
        chk_path = CHUNKS_V2_DIR / f"{ticker}.json"
        if chk_path.exists() and not args.force:
            try:
                if chk_path.stat().st_size>1000:
                    cnt=len(json.loads(chk_path.read_text()))
                    if cnt>0:
                        print(f"[{idx+1}/{len(subset)}] {ticker} SKIPPED cached chunks={cnt}")
                        # update existing
                        existing[ticker]=existing.get(ticker, {"ticker":ticker,"status":"ok","chunk_count":cnt})
                        ok+=1
                        total_chunks+=cnt
                        continue
            except:
                pass
        print(f"[{idx+1}/{len(subset)}] {ticker} {entry['cik']}")
        try:
            res=process_ticker(entry, filing_limits, delay=args.delay, force=args.force, force_sub_fetch=False)
            print(f"  -> {res['status']} chunks={res['chunk_count']}")
            existing[res["ticker"]]=res
            if res["status"]=="ok" or res["chunk_count"]>0:
                ok+=1
                total_chunks+=res["chunk_count"]
            else:
                fail+=1
        except Exception as e:
            print(f"  !! EXC {ticker} {e}")
            import traceback; traceback.print_exc()
            fail+=1
            existing[ticker]={"ticker":ticker,"status":f"exception:{e}","chunk_count":0}
        gc.collect()
        time.sleep(args.delay)
        if (idx+1)%20==0:
            # write interim manifest
            try:
                m_data={
                    "generated_at": __import__("datetime").datetime.utcnow().isoformat()+"Z",
                    "slice":{"start":args.start,"limit":args.limit,"processed_this_run":idx+1},
                    "totals":{"ok":ok,"fail":fail,"total_chunks":total_chunks,"total_entries":len(existing)},
                    "entries": list(existing.values())
                }
                manifest_path.write_text(json.dumps(m_data, indent=2))
                print(f"  interim manifest {manifest_path} ok={ok} fail={fail}")
            except Exception as e:
                print(f"manifest write fail {e}")
    # final manifest
    try:
        m_data={
            "generated_at": __import__("datetime").datetime.utcnow().isoformat()+"Z",
            "slice":{"start":args.start,"limit":args.limit,"processed_this_run":len(subset)},
            "totals":{"ok":ok,"fail":fail,"total_chunks":total_chunks,"total_entries":len(existing)},
            "entries": list(existing.values())
        }
        manifest_path.write_text(json.dumps(m_data, indent=2))
        print(f"DONE ok={ok} fail={fail} chunks={total_chunks}")
    except Exception as e:
        print(f"final manifest fail {e}")
    # chunks index
    try:
        import pathlib
        idx_path = CHUNKS_V2_DIR/"index.json"
        idx_data=[]
        for jf in CHUNKS_V2_DIR.glob("*.json"):
            if jf.name=="index.json":
                continue
            try:
                ch=json.loads(jf.read_text())
                idx_data.append({"ticker":jf.stem,"chunk_count":len(ch),"has_business":any(c.get("item_number")=="1" for c in ch),"has_risk":any(c.get("item_number")=="1A" for c in ch),"has_mda":any(c.get("item_number")=="7" for c in ch),"has_table":any(c.get("has_table") for c in ch)})
            except:
                continue
        idx_path.write_text(json.dumps(sorted(idx_data, key=lambda x:x["ticker"]), indent=2))
        print(f"index {len(idx_data)}")
    except Exception as e:
        print(f"index fail {e}")

if __name__=="__main__":
    main()
