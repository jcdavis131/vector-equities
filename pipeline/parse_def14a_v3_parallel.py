#!/usr/bin/env python3
"""Parallel wrapper for v3 fast parser using multiprocessing"""
import re, json, html, gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE_DEF = ROOT / "pipeline" / "cache" / "sec_def14a"
OUT_JSONL = ROOT / "pipeline" / "data" / "def14a_parsed_v3.jsonl"

# Import parse function from v3 module without re-importing BS heavy? We'll duplicate minimal logic here to avoid circular.
# Instead import parse_one_file_fast from parse_def14a_v3
sys.path.insert(0, str(ROOT / "pipeline"))
from parse_def14a_v3 import parse_one_file_fast

def parse_batch(file_list):
    results=[]
    for fp in file_list:
        try:
            r = parse_one_file_fast(Path(fp))
            results.append(r)
        except Exception as e:
            fp_path = Path(fp)
            results.append({"ticker": fp_path.name.split("_")[0], "filing_date": fp_path.name.split("_")[1] if "_" in fp_path.name else "", "file": f"pipeline/cache/sec_def14a/{fp_path.name}", "neo_count":0, "neos":[], "board_size":None, "candidates_found":0, "parse_success":False, "method": f"parallel-crash {e}"})
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    files = sorted(CACHE_DEF.glob("*.html"))
    print(f"Found {len(files)} html files", flush=True)
    if args.limit:
        files = files[:args.limit]
        print(f"Limited to {len(files)}", flush=True)

    already=set()
    if args.resume and OUT_JSONL.exists():
        try:
            with open(OUT_JSONL,"r") as f:
                for line in f:
                    try:
                        j=json.loads(line)
                        already.add(j.get("file",""))
                        already.add(Path(j.get("file","")).name)
                    except:
                        pass
            print(f"Resume already {len(already)}", flush=True)
            orig=len(files)
            files=[fl for fl in files if fl.name not in already and f"pipeline/cache/sec_def14a/{fl.name}" not in already]
            print(f"Remaining {len(files)} of {orig}", flush=True)
        except Exception as e:
            print(f"Resume fail {e}", flush=True)

    # split into chunks for workers
    workers=args.workers
    # create batches
    # For simplicity, use ProcessPoolExecutor mapping single files, not batches, to keep progress streaming
    # We'll use executor.submit for each file but with buffering

    mode="a" if args.resume else "w"
    out_f=open(OUT_JSONL, mode)
    total=0
    succ=0

    # Use as_completed for streaming
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(parse_one_file_fast, fp): fp for fp in files}
        for idx, future in enumerate(as_completed(future_to_file), 1):
            fp = future_to_file[future]
            try:
                r = future.result()
            except Exception as e:
                fp_path = Path(fp)
                r = {"ticker": fp_path.name.split("_")[0], "filing_date": fp_path.name.split("_")[1] if "_" in fp_path.name else "", "file": f"pipeline/cache/sec_def14a/{fp_path.name}", "neo_count":0, "neos":[], "board_size":None, "candidates_found":0, "parse_success":False, "method": f"parallel-crash {e}"}
            out_f.write(json.dumps(r)+"\n")
            out_f.flush()
            total+=1
            if r["parse_success"]:
                succ+=1
            if idx%25==0 or r["parse_success"]:
                names=[n["name"] for n in r["neos"][:3]]
                print(f"{idx}/{len(files)} {'OK' if r['parse_success'] else 'FAIL'} {Path(fp).name} neos {r['neo_count']} cand {r['candidates_found']} {r['method']} {names}", flush=True)
            if idx%100==0:
                gc.collect()

    out_f.close()
    # final stats
    try:
        with open(OUT_JSONL,"r") as fin:
            all_lines=[json.loads(l) for l in fin if l.strip()]
        succ=sum(1 for r in all_lines if r["parse_success"])
        total=len(all_lines)
        print(f"\nWrote {OUT_JSONL} total={total} succ={succ} rate={succ/total:.3f}", flush=True)
        from collections import Counter
        print("Methods:", Counter([r["method"] for r in all_lines if r["parse_success"]]).most_common(15), flush=True)
    except Exception as e:
        print(f"stats fail {e}", flush=True)
