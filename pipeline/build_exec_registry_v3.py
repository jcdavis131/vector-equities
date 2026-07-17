#!/usr/bin/env python3
"""
Build exec registry v3 from def14a_parsed_v3.jsonl

Normalization:
- lower, replace \xa0 strip, remove Jr/Sr/III/IV etc, collapse spaces
- key first+last = first token + last token
- timeline sorted by filing_date
- group

Detect moves: same norm_firstlast appears in >=2 tickers -> exec_registry_v3.json summary + exec_registry_moves.jsonl
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
IN_JSONL = ROOT / "pipeline" / "data" / "def14a_parsed_v3.jsonl"
OUT_REGISTRY = ROOT / "pipeline" / "data" / "exec_registry_v3.json"
OUT_MOVES = ROOT / "pipeline" / "data" / "exec_registry_moves.jsonl"

SUFFIXES = {"jr","sr","ii","iii","iv","jr.","sr."}

def normalize_full(name: str) -> str:
    if not name:
        return ""
    s=name.lower()
    s=s.replace("\xa0"," ").replace("\u00a0"," ")
    s=s.replace(","," ")
    s=re.sub(r"\s+"," ",s).strip()
    # Remove suffixes
    parts=s.split()
    # filter suffix tokens
    filtered=[]
    for p in parts:
        if p in SUFFIXES:
            continue
        # also handle "jr." etc already
        if p.rstrip(".") in SUFFIXES:
            continue
        filtered.append(p)
    s=" ".join(filtered)
    s=re.sub(r"\s+"," ",s).strip()
    return s

def normalize_firstlast(full_norm: str) -> str:
    if not full_norm:
        return ""
    parts=full_norm.split()
    if len(parts)<2:
        return full_norm
    first=parts[0]
    last=parts[-1]
    return f"{first} {last}"

def load_parsed():
    entries=[]
    if not IN_JSONL.exists():
        print(f"Input {IN_JSONL} not found!")
        sys.exit(1)
    with open(IN_JSONL,"r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                j=json.loads(line)
                entries.append(j)
            except:
                continue
    return entries

def main():
    entries=load_parsed()
    total_files=len(entries)
    succ=len([e for e in entries if e.get("parse_success")])
    print(f"Loaded {total_files} files, success {succ} rate {succ/total_files:.3f}")

    # Build registry dict: norm_firstlast -> {full_names set, tickers set, timeline list}
    registry=defaultdict(lambda: {"full_names": set(), "tickers": set(), "timeline": []})

    unique_execs_full=set()
    for e in entries:
        if not e.get("parse_success"):
            continue
        ticker=e.get("ticker","")
        filing_date=e.get("filing_date","")
        file_path=e.get("file","")
        for neo in e.get("neos",[]):
            full_name=neo.get("name","").strip()
            if not full_name or len(full_name)<3:
                continue
            unique_execs_full.add(full_name)
            full_norm=normalize_full(full_name)
            if not full_norm:
                continue
            fl_key=normalize_firstlast(full_norm)
            if not fl_key or len(fl_key.split())<2:
                continue
            role=neo.get("role")
            comp=neo.get("total_comp")
            registry[fl_key]["full_names"].add(full_name)
            registry[fl_key]["tickers"].add(ticker)
            registry[fl_key]["timeline"].append({
                "ticker": ticker,
                "filing_date": filing_date,
                "comp": comp,
                "role": role,
                "file": file_path,
                "full_name": full_name,
                "full_norm": full_norm
            })

    # Sort timelines
    for k,v in registry.items():
        # sort by filing_date
        v["timeline"] = sorted(v["timeline"], key=lambda x: x["filing_date"])
        v["full_names"]=sorted(list(v["full_names"]))
        v["tickers"]=sorted(list(v["tickers"]))

    unique_firstlast=len(registry)
    print(f"Unique execs full: {len(unique_execs_full)} unique firstlast: {unique_firstlast}")

    # Detect moves: same firstlast appears in >=2 tickers
    moves=[]
    for fl_key, data in registry.items():
        if len(data["tickers"])>=2:
            moves.append({
                "norm_firstlast": fl_key,
                "full_names": data["full_names"],
                "tickers": data["tickers"],
                "timeline": data["timeline"]
            })
    moves_sorted=sorted(moves, key=lambda x: len(x["tickers"]), reverse=True)

    print(f"Moves count: {len(moves_sorted)}")
    if moves_sorted:
        print("Examples moves:")
        for m in moves_sorted[:10]:
            print(f"  {m['norm_firstlast']} tickers={m['tickers']} names={m['full_names']} timeline_len={len(m['timeline'])}")

    # Write registry json (summary)
    # For registry json, we want mapping firstlast -> {full_names, tickers, count, first_seen, last_seen}
    registry_json={}
    for fl_key, data in registry.items():
        timeline=data["timeline"]
        if timeline:
            first_seen=min(t["filing_date"] for t in timeline if t["filing_date"])
            last_seen=max(t["filing_date"] for t in timeline if t["filing_date"])
        else:
            first_seen=""
            last_seen=""
        registry_json[fl_key]={
            "full_names": data["full_names"],
            "tickers": data["tickers"],
            "count": len(timeline),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "timeline": timeline  # full timeline inside?
        }

    # Write files
    OUT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REGISTRY,"w") as out:
        json.dump(registry_json, out, indent=2)
    print(f"Wrote {OUT_REGISTRY} entries {len(registry_json)}")

    with open(OUT_MOVES,"w") as out:
        for m in moves_sorted:
            out.write(json.dumps(m)+"\n")
    print(f"Wrote {OUT_MOVES} moves {len(moves_sorted)}")

    # Print summary for task
    print("\n=== SUMMARY ===")
    print(f"total_files: {total_files}")
    print(f"success_rate: {succ/total_files:.3f} ({succ}/{total_files})")
    print(f"unique_execs_full: {len(unique_execs_full)}")
    print(f"unique_firstlast: {unique_firstlast}")
    print(f"moves_count: {len(moves_sorted)}")
    if moves_sorted:
        print("examples:")
        for ex in moves_sorted[:5]:
            print(f"  {ex['norm_firstlast']} -> {ex['tickers']} timeline: {[(t['ticker'], t['filing_date'], t['comp'], t['role']) for t in ex['timeline'][:3]]}")

if __name__=="__main__":
    main()
