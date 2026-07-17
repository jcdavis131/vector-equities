#!/usr/bin/env python3
"""Robust chunked parser using subprocess isolation to avoid OOM"""
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"pipeline/cache/sec_def14a"
OUT=ROOT/"pipeline/data/def14a_parsed_v3.jsonl"

def load_already():
    already=set()
    if OUT.exists():
        for line in open(OUT, errors='ignore'):
            try:
                j=json.loads(line)
                already.add(Path(j['file']).name)
                already.add(j['file'].split('/')[-1])
            except:
                pass
    return already

def process_chunk(files):
    # files: list of Path
    # run in subprocess that imports parser and processes list
    file_list_str=",".join([str(f) for f in files])
    # we will write a temp python script
    script=f"""
import json, sys
from pathlib import Path
sys.path.insert(0, "pipeline")
from parse_def14a_v3 import parse_one_file_fast
files=[Path(p) for p in '''{file_list_str}'''.split(',') if p]
results=[]
for fp in files:
    try:
        r=parse_one_file_fast(fp)
        results.append(r)
    except Exception as e:
        results.append({{"ticker": fp.name.split('_')[0], "filing_date": fp.name.split('_')[1] if '_' in fp.name else "", "file": f"pipeline/cache/sec_def14a/{{fp.name}}", "neo_count":0, "neos":[], "board_size":None, "candidates_found":0, "parse_success": False, "method": f"crash {{e}}"}})
for r in results:
    print(json.dumps(r))
"""
    cmd=[sys.executable, "-c", script]
    # Actually need to handle file list with quotes better: use JSON encoding
    import json as js
    file_json=js.dumps([str(f) for f in files])
    script2=f"""
import json, sys
from pathlib import Path
sys.path.insert(0, "pipeline")
from parse_def14a_v3 import parse_one_file_fast
files=[Path(p) for p in json.loads('''{file_json}''')]
for fp in files:
    try:
        r=parse_one_file_fast(fp)
        print(json.dumps(r), flush=True)
    except Exception as e:
        r={{"ticker": fp.name.split('_')[0], "filing_date": fp.name.split('_')[1] if '_' in fp.name else "", "file": f"pipeline/cache/sec_def14a/{{fp.name}}", "neo_count":0, "neos":[], "board_size":None, "candidates_found":0, "parse_success": False, "method": f"crash {{e}}"}} 
        print(json.dumps(r), flush=True)
"""
    cmd=[sys.executable, "-c", script2]
    proc=subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    if proc.returncode!=0:
        print(f"Chunk failed returncode {proc.returncode} stderr {proc.stderr[:1000]}", file=sys.stderr)
        # fallback per file
        return []
    out_lines=[]
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            out_lines.append(json.loads(line))
        except:
            print(f"Bad line {line[:200]}", file=sys.stderr)
    return out_lines

def main():
    already=load_already()
    files=sorted(CACHE.glob("*.html"))
    remaining=[f for f in files if f.name not in already]
    print(f"Found {len(files)} total, already {len(already)} remaining {len(remaining)}")
    chunk_size=20
    total=0
    succ=0
    # open output in append
    out_f=open(OUT,"a")
    for i in range(0, len(remaining), chunk_size):
        chunk=remaining[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size+1}/{(len(remaining)+chunk_size-1)//chunk_size} files {len(chunk)}", flush=True)
        results=process_chunk(chunk)
        for r in results:
            out_f.write(json.dumps(r)+"\n")
            out_f.flush()
            total+=1
            if r.get('parse_success'):
                succ+=1
        print(f"Chunk done total processed this run {total} succ {succ} rate {succ/total if total else 0:.3f}", flush=True)
        # early stop if we reached target 804
        # check overall
        overall=len(already)+total
        if overall>=804:
            print(f"Reached target 804 (overall {overall}), stopping early")
            break
    out_f.close()
    # final stats
    with open(OUT) as f:
        lines=[json.loads(l) for l in f if l.strip()]
    succ_all=sum(1 for l in lines if l['parse_success'])
    print(f"Final {OUT} total={len(lines)} succ={succ_all} rate={succ_all/len(lines):.3f}")

if __name__=="__main__":
    main()
