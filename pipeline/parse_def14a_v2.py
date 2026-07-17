"""Improved DEF14A parser using BeautifulSoup handling iXBRL"""
import re, json, sys
from pathlib import Path
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
CACHE_DEF=ROOT/"pipeline"/"cache"/"sec_def14a"

def parse_one(html_path):
    try:
        raw=html_path.read_bytes().decode('utf-8', errors='ignore')
    except:
        return {"error":"read fail"}
    ticker=html_path.name.split('_')[0]
    fdate=html_path.name.split('_')[1] if '_' in html_path.name else ""
    
    soup=BeautifulSoup(raw, 'html.parser')
    tables=soup.find_all('table')
    # Find candidate tables: those containing "Summary Compensation" or headers with Salary/Bonus/Stock Awards
    candidates=[]
    for tbl in tables:
        txt=tbl.get_text(separator=' ', strip=True)
        low=txt.lower()
        if 'summary compensation table' in low:
            candidates.append((tbl, 'title match'))
        elif ('salary' in low and 'bonus' in low and 'total' in low) and len(txt)>200 and len(txt)<20000:
            # Likely SCT
            # Additional filter: contains name pattern
            candidates.append((tbl, 'salary+bonus+total'))
    # Also try div-based? For now focus on candidates
    neos=[]
    for tbl, reason in candidates[:3]:  # try first few
        rows=tbl.find_all('tr')
        if len(rows)<2:
            continue
        # Extract header row
        # Try to find header with Name and Principal Position
        header_text=' '.join([c.get_text(strip=True) for c in rows[0].find_all(['th','td'])])
        # Now parse following rows up to maybe 10
        for r in rows[1:12]:
            cells=r.find_all(['td','th'])
            if len(cells)<3:
                continue
            texts=[c.get_text(separator=' ', strip=True) for c in cells]
            if not texts:
                continue
            first=texts[0]
            if len(first)<3 or len(first)>100:
                continue
            # filter: first cell should contain 2+ words and not be dollar or year
            if re.match(r'^\$|^\d{4}$|^\d+\s*$', first):
                continue
            if first.lower() in ['name and principal position', 'name', 'total', '']:
                continue
            # Heuristic: look for CEO/CFO or typical exec names (capitalized first last)
            # Must have dollar amounts in row
            row_text=' '.join(texts)
            dollars=re.findall(r'\$\s*[\d,]+', row_text)
            if len(dollars)<1:
                # try without $ but large numbers
                nums=re.findall(r'[\d,]{4,}', row_text)
                if len(nums)<2:
                    continue
            # Attempt to extract total comp as last numeric
            # Clean first as name, remove title in same cell often includes title after name
            # Name pattern: up to 30 chars starting with capital
            # Split first cell by newline or double space
            name=first
            # Remove footnotes like (1) (2)
            name=re.sub(r'\(\d+\)', '', name).strip()
            # If contains title like 'Chairman', split
            # Simple: take first 2-3 words as name if rest looks like title?
            # For now keep full first cell but truncate after 40 chars
            if len(name.split())>=2 and len(name.split())<=5:
                # plausible name
                # extract comp: last number
                comp=None
                for t in reversed(texts):
                    m=re.search(r'([\d,]{3,})', t)
                    if m:
                        try:
                            val=m.group(1).replace(',','')
                            comp=float(val)
                            if comp>100000:  # threshold
                                break
                            else:
                                comp=None
                        except:
                            pass
                if comp and comp>500000:
                    neos.append({"name": name[:60].strip(), "row_text": row_text[:200], "total_comp": comp, "reason": reason})
        if len(neos)>=2:
            break
    # Dedupe
    seen=set()
    uniq=[]
    for n in neos:
        key=n["name"].lower().strip()
        if key not in seen and len(key)>3:
            seen.add(key)
            uniq.append(n)
            if len(uniq)>=6:
                break
    # Board size heuristic via text search
    full_text=soup.get_text(separator=' ', strip=True)[:20000]
    board_size=None
    m=re.search(r'board.*consists of (\d+)', full_text, re.IGNORECASE)
    if m:
        try:
            bs=int(m.group(1))
            if 3<=bs<=20:
                board_size=bs
        except:
            pass
    result={
        "ticker": ticker,
        "filing_date": fdate,
        "file": str(html_path),
        "neo_count": len(uniq),
        "neos": uniq,
        "board_size": board_size,
        "candidates_found": len(candidates),
        "parse_success": len(uniq)>=2,
    }
    return result

if __name__=="__main__":
    cache=Path("pipeline/cache/sec_def14a")
    files=sorted(cache.glob("*.html"))
    print(f"Found {len(files)} files")
    results=[]
    for f in files:
        r=parse_one(f)
        results.append(r)
        if results.__len__() % 20 ==0:
            print(f"{len(results)}/{len(files)} {f.name} neos {r['neo_count']} cand {r['candidates_found']}")
        else:
            # print only successes
            if r['parse_success']:
                print(f"OK {f.name} neos {r['neo_count']} { [n['name'] for n in r['neos'][:2]] }")
    out=Path("pipeline/data/def14a_parsed_v2.jsonl")
    with open(out,'w') as outf:
        for r in results:
            outf.write(json.dumps(r)+"\n")
    succ=sum(1 for r in results if r['parse_success'])
    print(f"Wrote {out} {len(results)} succ {succ} rate {succ/len(results):.2f}")
