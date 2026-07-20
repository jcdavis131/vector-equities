#!/usr/bin/env python3
"""
V2 SEC Wiki Ingestor — HTML-aware chunking, rank-ordered ingestion (guide Part 1-3)
"""
import argparse, json, re, time, subprocess, sys, gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
DATA_DIR = PIPELINE / "data"
CACHE_ROOT = PIPELINE / "cache"

SEC_SUBMISSIONS_DIR = CACHE_ROOT / "sec_submissions"
SEC_FED_CACHE = CACHE_ROOT / "sec"
SEC_10K_CACHE = CACHE_ROOT / "sec_10k"
SEC_10K_EXTRACTED_OLD = SEC_10K_CACHE / "extracted"
SEC_10K_EXTRACTED_V2 = SEC_10K_CACHE / "extracted" / "v2"
SEC_EDGAR_HTML = CACHE_ROOT / "sec_edgar_html"
CHUNKS_V2_DIR = DATA_DIR / "chunks_v2"

for p in [SEC_SUBMISSIONS_DIR, SEC_10K_EXTRACTED_OLD, SEC_10K_EXTRACTED_V2, SEC_EDGAR_HTML, CHUNKS_V2_DIR]:
    p.mkdir(parents=True, exist_ok=True)

USER_AGENT = "VectorEquities/1.0 (jcdavis131@gmail.com)"

try:
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

def find_universe_file() -> Path:
    candidates = [DATA_DIR / "universe_nasdaq_nyse_common_cik.json", DATA_DIR / "universe_full.json", DATA_DIR / "universe.json"]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1000:
            return p
    raise FileNotFoundError(f"No universe file found")

def load_universe(path: Optional[Path] = None) -> List[Dict]:
    if not path:
        path = find_universe_file()
    data = json.loads(path.read_text())
    supplement_path = DATA_DIR / "universe.json"
    supplement = []
    if supplement_path.exists():
        try:
            supplement = json.loads(supplement_path.read_text())
        except:
            pass
    out=[]
    seen=set()
    def norm(e):
        ticker = e.get("ticker") or e.get("symbol")
        cik = e.get("cik") or ""
        if not ticker: return None
        cik_str=re.sub(r'\D','',str(cik)).zfill(10) if re.sub(r'\D','',str(cik)) else ""
        return {"ticker":ticker.upper(),"ticker_sec":e.get("ticker_sec") or ticker.upper(),"cik":cik_str,"cik_nz":str(int(cik_str)) if cik_str and cik_str.isdigit() else "","company":e.get("company") or e.get("title") or ticker,"sector":e.get("sector") or "n/a","industry":e.get("industry") or "n/a","exchange":e.get("exchange") or "","raw":e}
    for e in data:
        n=norm(e)
        if not n or not n["cik"] or n["ticker"] in seen: continue
        seen.add(n["ticker"]); out.append(n)
    for e in supplement:
        n=norm(e)
        if not n or not n["cik"] or n["ticker"] in seen: continue
        if n["ticker"]=="FB": n["ticker"]="META"; n["cik"]="0001326801"; n["cik_nz"]="1326801"
        if n["ticker"]=="META": n["cik"]="0001326801"; n["cik_nz"]="1326801"
        seen.add(n["ticker"]); out.append(n)
    if "META" not in seen:
        out.append({"ticker":"META","ticker_sec":"META","cik":"0001326801","cik_nz":"1326801","company":"Meta Platforms Inc","sector":"Technology","industry":"Internet","exchange":"NASDAQ","raw":{"ticker":"META"}})
    return out

def robust_fetch_json_curl(url: str, timeout: int = 90) -> Optional[Dict]:
    tmp=Path(f"/tmp/sec_json_{int(time.time()*1000)}.json")
    cmd=["curl","-sL","--compressed","--max-time",str(timeout),"-H",f"User-Agent: {USER_AGENT}","-H","Accept: application/json",url,"-o",str(tmp),"-w","%{http_code}"]
    try:
        subprocess.run(cmd,capture_output=True,text=True,timeout=timeout+10)
        if tmp.exists() and tmp.stat().st_size>100:
            try:
                data=json.loads(tmp.read_text())
                tmp.unlink(missing_ok=True)
                return data
            except:
                try:
                    return json.loads(tmp.read_text())
                except:
                    tmp.unlink(missing_ok=True)
                    return None
    except: pass
    return None

def fetch_submission_if_missing(cik_pad: str, delay: float = 0.25) -> Optional[Dict]:
    sub_path=SEC_SUBMISSIONS_DIR / f"sub_{cik_pad}.json"
    if sub_path.exists() and sub_path.stat().st_size>2000:
        try: return json.loads(sub_path.read_text())
        except: sub_path.unlink(missing_ok=True)
    url=f"https://data.sec.gov/submissions/CIK{cik_pad}.json"
    data=robust_fetch_json_curl(url)
    if data:
        try: sub_path.write_text(json.dumps(data))
        except: pass
        time.sleep(delay)
        return data
    time.sleep(delay+0.5)
    return None

def get_filings_by_type(sub_data: Dict, target_forms: List[str], disallowed: List[str]) -> Dict[str, List[Dict]]:
    filings=sub_data.get("filings",{}) if isinstance(sub_data.get("filings"),dict) else {}
    recent=filings.get("recent",{}) if isinstance(filings,dict) else {}
    forms=recent.get("form",[]); accessions=recent.get("accessionNumber",[]); prim_docs=recent.get("primaryDocument",[]); filing_dates=recent.get("filingDate",[]); report_dates=recent.get("reportDate",[])
    all_filings=[]
    for i in range(min(len(forms),len(accessions))):
        f=forms[i]
        if not f: continue
        f_up=f.upper().strip()
        if f_up in [d.upper() for d in disallowed] or f_up in ["3","4","5"]: continue
        if f_up.startswith("13") or "13D" in f_up or "13G" in f_up or f_up in ["SC 13D","SC 13G","SC 13D/A","SC 13G/A"]: continue
        acc=accessions[i] if i<len(accessions) else ""; pd=prim_docs[i] if i<len(prim_docs) else ""; fd=filing_dates[i] if i<len(filing_dates) else ""; rd=report_dates[i] if i<len(report_dates) else ""
        if not acc or not pd: continue
        all_filings.append({"form":f,"accessionNumber":acc,"primaryDocument":pd,"filingDate":fd,"reportDate":rd})
    def sort_key(x): return x.get("filingDate") or "0000"
    all_filings_sorted=sorted(all_filings,key=sort_key,reverse=True)
    grouped={tf:[] for tf in target_forms}
    for fil in all_filings_sorted:
        ft=fil["form"]
        if ft in grouped: grouped[ft].append(fil)
        if ft in ["DEF 14A","DEF14A"]:
            grouped.setdefault("DEF14A",[]).append(fil) if fil not in grouped.get("DEF14A",[]) else None
    if "DEF 14A" in grouped and "DEF14A" in target_forms:
        for f in grouped["DEF 14A"]:
            if f not in grouped["DEF14A"]: grouped["DEF14A"].append(f)
    return grouped

def get_latest_filings(sub_data: Dict, limit_per_type: Dict[str,int]) -> List[Dict]:
    target_forms=list(limit_per_type.keys())
    disallowed=["3","4","5","13D","13G","SC 13D","SC 13G","SC 13D/A","SC 13G/A","13F-HR","D","1-A"]
    grouped=get_filings_by_type(sub_data,target_forms,disallowed)
    out=[]
    for form,lim in limit_per_type.items():
        candidates=grouped.get(form,[])
        if form=="DEF14A":
            candidates=candidates+grouped.get("DEF 14A",[])
            seen=set(); uniq=[]
            for c in candidates:
                acc=c.get("accessionNumber")
                if acc not in seen: seen.add(acc); uniq.append(c)
            candidates=uniq
        for c in candidates[:lim]:
            c_copy=dict(c); c_copy["requested_form"]=form; out.append(c_copy)
    return out

def fetch_edgar_html(cik_nz: str, accession_nodash: str, primary_doc: str, dest_path: Path, delay: float = 0.3, force: bool = False) -> Tuple[bool,str]:
    if dest_path.exists() and dest_path.stat().st_size>1000 and not force:
        try:
            with open(dest_path,'rb') as f: magic=f.read(3)
            if magic!=b'\x1f\x8b\x08': return True,"cached"
        except: return True,"cached"
    dest_path.parent.mkdir(parents=True,exist_ok=True)
    url=f"https://www.sec.gov/Archives/edgar/data/{cik_nz}/{accession_nodash}/{primary_doc}"
    tmp_path=Path(f"/tmp/edgar_{cik_nz}_{accession_nodash}_{primary_doc.replace('/','_')}.html")
    try:
        if tmp_path.exists(): tmp_path.unlink()
    except: pass
    cmd=["curl","-sL","--compressed","--max-time","120","-H",f"User-Agent: {USER_AGENT}","-H","Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","-H","Host: www.sec.gov",url,"-o",str(tmp_path),"-w","%{http_code}"]
    try:
        res=subprocess.run(cmd,capture_output=True,text=True,timeout=130)
        http_code=res.stdout.strip()[-3:] if res.stdout else "000"
        if tmp_path.exists() and tmp_path.stat().st_size>1000:
            try: tmp_path.replace(dest_path)
            except:
                dest_path.write_bytes(tmp_path.read_bytes()); tmp_path.unlink(missing_ok=True)
            time.sleep(delay)
            return True,http_code
        else:
            sz=tmp_path.stat().st_size if tmp_path.exists() else 0
            time.sleep(delay+0.5)
            return False,f"curl_code={http_code} size={sz}"
    except Exception as e:
        time.sleep(delay+1)
        return False,f"exception {e}"

BOILERPLATE_PHRASES=["forward-looking statements","cautionary statement regarding","table of contents","pursuant to the requirements of the securities exchange act","signatures","exhibit index","index to financial statements","conformed submission","documents incorporated by reference"]

def is_boilerplate(text: str) -> bool:
    tl=text.lower()
    substance=["revenue","product","customer","market","competitor","growth","service","platform","we believe","our strategy"]
    for ph in BOILERPLATE_PHRASES:
        if ph in tl:
            if len(tl)<800: return True
            if sum(1 for kw in substance if kw in tl)<2 and len(tl)<1200: return True
    if len(tl)<80 and tl.strip().endswith("..."): return True
    return False

ITEM_DETECTION_REGEX=re.compile(r'(?i)^\s*ITEM\s+([0-9]+[A-C]?)\b[^\n]{0,120}',re.MULTILINE)

def detect_item_id(text: str) -> Optional[str]:
    if len(text)>300: snippet=text[:300]
    else: snippet=text
    m=ITEM_DETECTION_REGEX.search(snippet)
    if not m: return None
    raw=m.group(1).upper().strip()
    if not re.match(r'^[0-9]+[A-C]?$',raw): return None
    return raw

def is_subheader_candidate(text: str, tag_name: str) -> bool:
    txt=text.strip()
    if len(txt)<3 or len(txt)>200: return False
    if txt.endswith('.') and len(txt)>80: return False
    if tag_name in ['h1','h2','h3','h4','b','strong']:
        if detect_item_id(txt): return False
        if len(txt)>5: return True
    if txt.isupper() and len(txt.split())<=12 and len(txt)>=8: return True
    if len(txt.split())<=10 and not txt.endswith('.') and txt[0].isupper():
        if len(txt)<100: return True
    return False

def table_to_markdown(table_tag) -> str:
    try:
        rows=table_tag.find_all('tr')
        if not rows or len(rows)>100: rows=rows[:30]
        md_lines=[]
        for i,tr in enumerate(rows):
            cells=tr.find_all(['th','td'])
            if not cells: continue
            cell_texts=[]
            for cell in cells[:12]:
                ct=cell.get_text(separator=' ',strip=True).replace('\n',' ').replace('|',' ').strip()
                if len(ct)>150: ct=ct[:150]+"..."
                cell_texts.append(ct)
            if not cell_texts or all(not c for c in cell_texts): continue
            if len([c for c in cell_texts if c])<2: continue
            md_lines.append("| "+" | ".join(cell_texts)+" |")
            if i==0: md_lines.append("| "+" | ".join(["---"]*len(cell_texts))+" |")
            if len(md_lines)>25:
                md_lines.append("| ... truncated ... |"); break
        if not md_lines: return ""
        return "\n".join(md_lines)
    except Exception as e:
        return f"<!-- table parse error {e} -->"

def parse_10k_html_v2(html_content: str, ticker_meta: Dict) -> Dict[str, List[Dict]]:
    if not HAS_BS4: raise ImportError("BeautifulSoup required")
    soup=BeautifulSoup(html_content,"lxml")
    for s in soup(["script","style","noscript"]): s.decompose()
    body=soup.body or soup
    relevant_tags=body.find_all(['p','div','table','h1','h2','h3','h4','b','strong','li','span','font','center'])
    marker_indices=[]
    for idx,tag in enumerate(relevant_tags):
        try: t=tag.get_text(separator=' ',strip=True)
        except: continue
        if not t or len(t)<4 or len(t)>400: continue
        iid=detect_item_id(t)
        if iid: marker_indices.append((idx,iid))
    last_occurrence={}
    for idx,iid in marker_indices: last_occurrence[iid]=idx
    sorted_last=sorted([(idx,iid) for iid,idx in last_occurrence.items()],key=lambda x:x[0])
    iid_to_sorted_pos={iid:pos for pos,(idx,iid) in enumerate(sorted_last)}
    if not sorted_last:
        return {k:[] for k in ["1","1A","1B","1C","2","3","5","6","7","7A","8","9"]}
    boundaries={}
    for target in ["1","1A","7"]:
        if target not in last_occurrence: continue
        start_idx=last_occurrence[target]
        cur_pos=iid_to_sorted_pos.get(target,-1)
        if cur_pos==-1: continue
        if cur_pos+1 < len(sorted_last): end_idx=sorted_last[cur_pos+1][0]
        else: end_idx=min(len(relevant_tags), start_idx+8000)
        boundaries[target]=(start_idx,end_idx)
    buffers={k:[] for k in ["1","1A","1B","1C","2","3","5","6","7","7A","8","9"]}
    def process_range(target_id,start_idx,end_idx):
        last_subheader=f"Item {target_id}"
        cur_buff=buffers[target_id]
        for ridx in range(start_idx+1,end_idx):
            if ridx>=len(relevant_tags): break
            tag=relevant_tags[ridx]
            try:
                parent=tag.parent
                inside_table=False
                for _ in range(3):
                    if parent is None: break
                    if getattr(parent,'name',None)=='table': inside_table=True; break
                    parent=getattr(parent,'parent',None)
                if inside_table and tag.name!='table': continue
            except: pass
            if tag.name=='table':
                md=table_to_markdown(tag)
                if md and len(md)>20:
                    cur_buff.append({"type":"table","text":md,"header":last_subheader,"tag":"table"})
                continue
            try: txt_raw=tag.get_text(separator=' ',strip=True)
            except: continue
            if not txt_raw or len(txt_raw)<30:
                if len(txt_raw)>=3 and len(txt_raw)<=200 and is_subheader_candidate(txt_raw,tag.name):
                    last_subheader=txt_raw
                    cur_buff.append({"type":"subheader","text":txt_raw,"tag":tag.name})
                continue
            if len(txt_raw)<350:
                iid=detect_item_id(txt_raw)
                if iid and iid!=target_id: break
            if is_boilerplate(txt_raw): continue
            if is_subheader_candidate(txt_raw,tag.name):
                last_subheader=txt_raw
                cur_buff.append({"type":"subheader","text":txt_raw,"tag":tag.name})
                continue
            if cur_buff and cur_buff[-1].get("type")=="para" and cur_buff[-1].get("text")==txt_raw: continue
            cur_buff.append({"type":"para","text":txt_raw,"header":last_subheader,"tag":tag.name})
            if len(cur_buff)>2500: break
    for tgt,(s,e) in boundaries.items(): process_range(tgt,s,e)
    if not boundaries and last_occurrence:
        min_idx=min(last_occurrence.values())
        generic_end=min(len(relevant_tags),min_idx+8000)
        for tgt in ["1","1A","7"]:
            if tgt in last_occurrence:
                s=last_occurrence[tgt]
                next_idx=generic_end
                for m_idx,m_iid in sorted_last:
                    if m_idx> s: next_idx=m_idx; break
                process_range(tgt,s,next_idx)
    # explicit cleanup
    try: del soup, relevant_tags
    except: pass
    gc.collect()
    return buffers

def chunk_from_buffers(buffers: Dict[str, List[Dict]], ticker_meta: Dict, filing_meta: Dict, target_tokens: int = 750) -> List[Dict]:
    chunks=[]
    ticker=ticker_meta["ticker"]; form=filing_meta["form"]; year=filing_meta.get("filing_year") or "2024"; quarter=filing_meta.get("filing_quarter")
    cik=ticker_meta.get("cik") or ""; company=ticker_meta.get("company") or ticker; acc=filing_meta.get("accessionNumber") or ""; fdate=filing_meta.get("filingDate") or ""; surl=filing_meta.get("source_url") or ""
    global_idx=0
    for item_number in ["1","1A","7"]:
        elems=buffers.get(item_number,[])
        if not elems: continue
        cur_text=""; cur_header=""; last_header=""
        for elem in elems:
            et=elem.get("type")
            if et=="subheader":
                last_header=elem.get("text") or ""
                if not cur_text: cur_header=last_header
                continue
            if et=="table":
                if cur_text.strip() and len(cur_text)>200:
                    cid=f"{ticker}_{form}_{item_number}_{year}_{global_idx}"
                    chunk={"id":cid,"company_ticker":ticker,"ticker":ticker,"filing_type":form,"filing_year":int(year) if str(year).isdigit() else year,"filing_quarter":quarter or ("FY" if form=="10-K" else None),"item_number":item_number,"section_header":cur_header or elem.get("header") or last_header,"text":cur_text.strip(),"token_count":len(cur_text)//4,"source_url":surl,"accessionNumber":acc,"filingDate":fdate,"cik":cik,"company":company,"chunk_index":global_idx,"has_table":False,"metadata":{"company_ticker":ticker,"filing_type":form,"filing_year":year,"filing_quarter":quarter or "FY","item_number":item_number,"section_header":cur_header or elem.get("header") or last_header}}
                    if not is_boilerplate(chunk["text"]) and len(chunk["text"])>200:
                        chunks.append(chunk); global_idx+=1
                    cur_text=""; cur_header=last_header or cur_header
                table_md=elem.get("text") or ""
                injected=f"[Table Context: {ticker} {form} Item {item_number} — {cur_header or elem.get('header') or 'Financial/Operational Table'}]\n{table_md}"
                cid=f"{ticker}_{form}_{item_number}_{year}_{global_idx}_table"
                chunk={"id":cid,"company_ticker":ticker,"ticker":ticker,"filing_type":form,"filing_year":int(year) if str(year).isdigit() else year,"filing_quarter":quarter or ("FY" if form=="10-K" else None),"item_number":item_number,"section_header":cur_header or elem.get("header") or last_header,"text":injected,"token_count":len(injected)//4,"source_url":surl,"accessionNumber":acc,"filingDate":fdate,"cik":cik,"company":company,"chunk_index":global_idx,"has_table":True,"metadata":{"company_ticker":ticker,"filing_type":form,"filing_year":year,"filing_quarter":quarter or "FY","item_number":item_number,"section_header":cur_header or elem.get("header") or last_header}}
                chunks.append(chunk); global_idx+=1
                continue
            if et=="para":
                txt=elem.get("text") or ""
                if not cur_header: cur_header=elem.get("header") or last_header or ""
                if cur_text: cur_text+="\n\n"
                cur_text+=txt
                if len(cur_text)//4 >= target_tokens:
                    cid=f"{ticker}_{form}_{item_number}_{year}_{global_idx}"
                    chunk={"id":cid,"company_ticker":ticker,"ticker":ticker,"filing_type":form,"filing_year":int(year) if str(year).isdigit() else year,"filing_quarter":quarter or ("FY" if form=="10-K" else None),"item_number":item_number,"section_header":cur_header or last_header,"text":cur_text.strip(),"token_count":len(cur_text)//4,"source_url":surl,"accessionNumber":acc,"filingDate":fdate,"cik":cik,"company":company,"chunk_index":global_idx,"has_table":False,"metadata":{"company_ticker":ticker,"filing_type":form,"filing_year":year,"filing_quarter":quarter or "FY","item_number":item_number,"section_header":cur_header or last_header}}
                    if not is_boilerplate(chunk["text"]) and len(chunk["text"])>200:
                        chunks.append(chunk); global_idx+=1
                    cur_text=""; cur_header=last_header or cur_header
        if cur_text.strip():
            if chunks and chunks[-1]["item_number"]==item_number and not chunks[-1].get("has_table") and len(cur_text)//4 < 300:
                chunks[-1]["text"]=chunks[-1]["text"]+"\n\n"+cur_text.strip()
                chunks[-1]["token_count"]=len(chunks[-1]["text"])//4
            else:
                cid=f"{ticker}_{form}_{item_number}_{year}_{global_idx}"
                chunk={"id":cid,"company_ticker":ticker,"ticker":ticker,"filing_type":form,"filing_year":int(year) if str(year).isdigit() else year,"filing_quarter":quarter or ("FY" if form=="10-K" else None),"item_number":item_number,"section_header":cur_header or last_header,"text":cur_text.strip(),"token_count":len(cur_text)//4,"source_url":surl,"accessionNumber":acc,"filingDate":fdate,"cik":cik,"company":company,"chunk_index":global_idx,"has_table":False,"metadata":{"company_ticker":ticker,"filing_type":form,"filing_year":year,"filing_quarter":quarter or "FY","item_number":item_number,"section_header":cur_header or last_header}}
                if not is_boilerplate(chunk["text"]) and len(chunk["text"])>200:
                    chunks.append(chunk); global_idx+=1
    return chunks

def parse_def14a_v2(html_content: str, ticker_meta: Dict, filing_meta: Dict) -> List[Dict]:
    if not HAS_BS4: return []
    soup=BeautifulSoup(html_content,"lxml")
    for s in soup(["script","style"]): s.decompose()
    text=soup.get_text(separator="\n")
    patterns={"BOARD":r"(?i)(CORPORATE GOVERNANCE|BOARD OF DIRECTORS|DIRECTOR INDEPENDENCE)","CDA":r"(?i)(COMPENSATION DISCUSSION AND ANALYSIS)","EXEC_COMP":r"(?i)(EXECUTIVE COMPENSATION|SUMMARY COMPENSATION TABLE)"}
    chunks=[]
    for sec_name,pat in patterns.items():
        m=re.search(pat,text)
        if not m: continue
        start=m.start()
        next_sec=len(text)
        for other_pat in patterns.values():
            if other_pat==pat: continue
            nxt=re.search(other_pat,text[start+500:])
            if nxt:
                cand=start+500+nxt.start()
                if cand<next_sec: next_sec=cand
        section_text=text[start:min(start+150000,next_sec)]
        paras=[p.strip() for p in section_text.split("\n\n") if len(p.strip())>100]
        cur=""
        for p in paras:
            cur+="\n\n"+p
            if len(cur)//4>800:
                chunk={"id":f"{ticker_meta['ticker']}_{filing_meta['form']}_{sec_name}_{filing_meta.get('filing_year')}_{len(chunks)}","company_ticker":ticker_meta['ticker'],"ticker":ticker_meta['ticker'],"filing_type":"DEF14A","filing_year":filing_meta.get("filing_year") or 2024,"filing_quarter":None,"item_number":sec_name,"section_header":sec_name,"text":cur.strip(),"token_count":len(cur)//4,"source_url":filing_meta.get("source_url",""),"accessionNumber":filing_meta.get("accessionNumber",""),"filingDate":filing_meta.get("filingDate",""),"cik":ticker_meta.get("cik",""),"company":ticker_meta.get("company",""),"chunk_index":len(chunks),"has_table":False,"metadata":{"company_ticker":ticker_meta['ticker'],"filing_type":"DEF14A","filing_year":filing_meta.get("filing_year"),"filing_quarter":None,"item_number":sec_name,"section_header":sec_name}}
                if len(cur)>200: chunks.append(chunk)
                cur=""
        if cur.strip():
            chunk={"id":f"{ticker_meta['ticker']}_{filing_meta['form']}_{sec_name}_{filing_meta.get('filing_year')}_{len(chunks)}","company_ticker":ticker_meta['ticker'],"ticker":ticker_meta['ticker'],"filing_type":"DEF14A","filing_year":filing_meta.get("filing_year") or 2024,"filing_quarter":None,"item_number":sec_name,"section_header":sec_name,"text":cur.strip(),"token_count":len(cur)//4,"source_url":filing_meta.get("source_url",""),"accessionNumber":filing_meta.get("accessionNumber",""),"filingDate":filing_meta.get("filingDate",""),"cik":ticker_meta.get("cik",""),"company":ticker_meta.get("company",""),"chunk_index":len(chunks),"has_table":False,"metadata":{"company_ticker":ticker_meta['ticker'],"filing_type":"DEF14A","filing_year":filing_meta.get("filing_year"),"filing_quarter":None,"item_number":sec_name,"section_header":sec_name}}
            chunks.append(chunk)
    try: del soup
    except: pass
    gc.collect()
    return chunks

def parse_10q_item2_v2(html_content: str, ticker_meta: Dict, filing_meta: Dict) -> List[Dict]:
    if not HAS_BS4: return []
    soup=BeautifulSoup(html_content,"lxml")
    for s in soup(["script","style"]): s.decompose()
    text=soup.get_text(separator="\n")
    m=re.search(r'(?i)ITEM\s+2\s*[\.\-\)]*\s*MANAGEMENT',text)
    if not m: return []
    start=m.start()
    next_pat=re.search(r'(?i)\n\s*ITEM\s+(3|4)\b',text[start+100:])
    end=start+100+next_pat.start() if next_pat else min(len(text),start+150000)
    section_text=text[start:end]
    paras=[p.strip() for p in section_text.split("\n\n") if len(p.strip())>80]
    chunks=[]; cur=""; cur_header="Item 2 - MD&A"
    for p in paras:
        if is_boilerplate(p): continue
        cur+="\n\n"+p
        if len(cur)//4>800:
            chunk={"id":f"{ticker_meta['ticker']}_10-Q_2_{filing_meta.get('filing_year')}_{len(chunks)}","company_ticker":ticker_meta['ticker'],"ticker":ticker_meta['ticker'],"filing_type":"10-Q","filing_year":filing_meta.get("filing_year") or 2024,"filing_quarter":filing_meta.get("filing_quarter") or "Q","item_number":"2","section_header":cur_header,"text":cur.strip(),"token_count":len(cur)//4,"source_url":filing_meta.get("source_url",""),"accessionNumber":filing_meta.get("accessionNumber",""),"filingDate":filing_meta.get("filingDate",""),"cik":ticker_meta.get("cik",""),"company":ticker_meta.get("company",""),"chunk_index":len(chunks),"has_table":False,"metadata":{"company_ticker":ticker_meta['ticker'],"filing_type":"10-Q","filing_year":filing_meta.get("filing_year"),"filing_quarter":filing_meta.get("filing_quarter"),"item_number":"2","section_header":cur_header}}
            chunks.append(chunk); cur=""
    if cur.strip():
        chunk={"id":f"{ticker_meta['ticker']}_10-Q_2_{filing_meta.get('filing_year')}_{len(chunks)}","company_ticker":ticker_meta['ticker'],"ticker":ticker_meta['ticker'],"filing_type":"10-Q","filing_year":filing_meta.get("filing_year") or 2024,"filing_quarter":filing_meta.get("filing_quarter") or "Q","item_number":"2","section_header":cur_header,"text":cur.strip(),"token_count":len(cur)//4,"source_url":filing_meta.get("source_url",""),"accessionNumber":filing_meta.get("accessionNumber",""),"filingDate":filing_meta.get("filingDate",""),"cik":ticker_meta.get("cik",""),"company":ticker_meta.get("company",""),"chunk_index":len(chunks),"has_table":False,"metadata":{"company_ticker":ticker_meta['ticker'],"filing_type":"10-Q","filing_year":filing_meta.get("filing_year"),"filing_quarter":filing_meta.get("filing_quarter"),"item_number":"2","section_header":cur_header}}
        chunks.append(chunk)
    try: del soup
    except: pass
    gc.collect()
    return chunks

def parse_8k_ex99_v2(html_content: str, ticker_meta: Dict, filing_meta: Dict) -> List[Dict]:
    if not HAS_BS4: return []
    soup=BeautifulSoup(html_content,"lxml")
    for s in soup(["script","style"]): s.decompose()
    text=soup.get_text(separator="\n")
    if "99.1" not in text and "Press Release" not in text:
        if "99.2" not in text: return []
    m=re.search(r'(?i)(EXHIBIT\s+99\.(1|2)|PRESS RELEASE)',text)
    if not m: return []
    start=m.start()
    section=text[start:min(len(text),start+120000)]
    paras=[p.strip() for p in section.split("\n\n") if len(p.strip())>80]
    chunks=[]; cur=""
    for p in paras:
        if is_boilerplate(p): continue
        cur+="\n\n"+p
        if len(cur)//4>800:
            chunk={"id":f"{ticker_meta['ticker']}_8-K_EX99_{filing_meta.get('filing_year')}_{len(chunks)}","company_ticker":ticker_meta['ticker'],"ticker":ticker_meta['ticker'],"filing_type":"8-K","filing_year":filing_meta.get("filing_year") or 2024,"filing_quarter":None,"item_number":"EX99","section_header":"Exhibit 99.1 Press Release","text":cur.strip(),"token_count":len(cur)//4,"source_url":filing_meta.get("source_url",""),"accessionNumber":filing_meta.get("accessionNumber",""),"filingDate":filing_meta.get("filingDate",""),"cik":ticker_meta.get("cik",""),"company":ticker_meta.get("company",""),"chunk_index":len(chunks),"has_table":False,"metadata":{"company_ticker":ticker_meta['ticker'],"filing_type":"8-K","filing_year":filing_meta.get("filing_year"),"filing_quarter":None,"item_number":"EX99","section_header":"Exhibit 99.1"}}
            chunks.append(chunk); cur=""
    if cur.strip():
        chunk={"id":f"{ticker_meta['ticker']}_8-K_EX99_{filing_meta.get('filing_year')}_{len(chunks)}","company_ticker":ticker_meta['ticker'],"ticker":ticker_meta['ticker'],"filing_type":"8-K","filing_year":filing_meta.get("filing_year") or 2024,"filing_quarter":None,"item_number":"EX99","section_header":"Exhibit 99.1 Press Release","text":cur.strip(),"token_count":len(cur)//4,"source_url":filing_meta.get("source_url",""),"accessionNumber":filing_meta.get("accessionNumber",""),"filingDate":filing_meta.get("filingDate",""),"cik":ticker_meta.get("cik",""),"company":ticker_meta.get("company",""),"chunk_index":len(chunks),"has_table":False,"metadata":{"company_ticker":ticker_meta['ticker'],"filing_type":"8-K","filing_year":filing_meta.get("filing_year"),"filing_quarter":None,"item_number":"EX99","section_header":"Exhibit 99.1"}}
        chunks.append(chunk)
    try: del soup
    except: pass
    gc.collect()
    return chunks

def derive_filing_year_quarter(report_date: str, filing_date: str, form: str):
    year="2024"
    if report_date and len(report_date)>=4: year=report_date[:4]
    elif filing_date and len(filing_date)>=4: year=filing_date[:4]
    quarter=None
    if form=="10-K": quarter="FY"
    elif form=="10-Q":
        try:
            if report_date:
                month=int(report_date.split("-")[1])
                if month<=3: quarter="Q1"
                elif month<=6: quarter="Q2"
                elif month<=9: quarter="Q3"
                else: quarter="Q4"
            else: quarter="Q"
        except: quarter="Q"
    return year,quarter

def process_ticker(entry: Dict, filing_limits: Dict[str,int], delay: float = 0.3, force: bool = False, force_sub_fetch: bool = False) -> Dict:
    ticker=entry["ticker"]; cik_pad=entry["cik"]; cik_nz=entry["cik_nz"]
    if not cik_pad: return {"ticker":ticker,"status":"no_cik","chunk_count":0,"has_business":False,"has_risk":False,"has_mda":False,"chunks_file":"","filings_fetched":[]}
    sub_data=None
    sub_path=SEC_SUBMISSIONS_DIR / f"sub_{cik_pad}.json"
    if force_sub_fetch or not sub_path.exists() or sub_path.stat().st_size<2000:
        sub_data=fetch_submission_if_missing(cik_pad,delay=delay)
    else:
        try: sub_data=json.loads(sub_path.read_text())
        except: sub_data=fetch_submission_if_missing(cik_pad,delay=delay)
    if not sub_data:
        return {"ticker":ticker,"status":"no_submissions","chunk_count":0,"has_business":False,"has_risk":False,"has_mda":False,"chunks_file":"","filings_fetched":[]}
    filings_to_fetch=get_latest_filings(sub_data,filing_limits)
    if not filings_to_fetch:
        return {"ticker":ticker,"status":"no_target_filings","chunk_count":0,"has_business":False,"has_risk":False,"has_mda":False,"chunks_file":"","filings_fetched":[]}
    all_chunks=[]; fetched_html_status=[]; business_text_accum=""; risk_text_accum=""; mda_text_accum=""
    for filing in filings_to_fetch:
        form=filing.get("requested_form") or filing.get("form"); acc=filing["accessionNumber"]; acc_nodash=acc.replace("-",""); primary_doc=filing["primaryDocument"]; filing_date=filing.get("filingDate",""); report_date=filing.get("reportDate","")
        filing_year,filing_quarter=derive_filing_year_quarter(report_date,filing_date,form)
        source_url=f"https://www.sec.gov/Archives/edgar/data/{cik_nz}/{acc_nodash}/{primary_doc}"
        html_dir=SEC_EDGAR_HTML / cik_pad; html_path=html_dir / f"{acc_nodash}_{primary_doc.replace('/','_')}.html"
        ok,code=fetch_edgar_html(cik_nz,acc_nodash,primary_doc,html_path,delay=delay,force=force)
        fetched_html_status.append({"form":form,"accession":acc,"primary_doc":primary_doc,"html_path":str(html_path),"status":"ok" if ok else "fail","code":code,"filing_year":filing_year,"filing_quarter":filing_quarter})
        if not ok: continue
        try: html_content=html_path.read_text(errors="ignore")
        except: continue
        filing_meta={"form":form,"accessionNumber":acc,"primaryDocument":primary_doc,"filingDate":filing_date,"reportDate":report_date,"filing_year":filing_year,"filing_quarter":filing_quarter,"year":filing_year,"source_url":source_url}
        ticker_meta={"ticker":ticker,"cik":cik_pad,"cik_nz":cik_nz,"company":entry.get("company") or ticker}
        try:
            if form=="10-K":
                buffers=parse_10k_html_v2(html_content,ticker_meta)
                chunks=chunk_from_buffers(buffers,ticker_meta,filing_meta,target_tokens=750)
                all_chunks.extend(chunks)
                for ch in chunks:
                    if ch["item_number"]=="1": business_text_accum+="\n\n"+ch["text"]
                    elif ch["item_number"]=="1A": risk_text_accum+="\n\n"+ch["text"]
                    elif ch["item_number"]=="7": mda_text_accum+="\n\n"+ch["text"]
            elif form in ["DEF14A","DEF 14A"]:
                all_chunks.extend(parse_def14a_v2(html_content,ticker_meta,filing_meta))
            elif form=="10-Q":
                all_chunks.extend(parse_10q_item2_v2(html_content,ticker_meta,filing_meta))
            elif form=="8-K":
                all_chunks.extend(parse_8k_ex99_v2(html_content,ticker_meta,filing_meta))
            else:
                buffers=parse_10k_html_v2(html_content,ticker_meta)
                all_chunks.extend(chunk_from_buffers(buffers,ticker_meta,filing_meta,target_tokens=750))
        except Exception as e:
            print(f"[{ticker}] parse error {form} {acc}: {e}",file=sys.stderr)
            import traceback; traceback.print_exc()
            continue
        # cleanup per filing
        try: del html_content
        except: pass
        gc.collect()
    out_path=CHUNKS_V2_DIR / f"{ticker}.json"
    try: out_path.write_text(json.dumps(all_chunks,indent=2),encoding="utf-8")
    except Exception as e: print(f"Failed to write chunks for {ticker}: {e}")
    if business_text_accum:
        (SEC_10K_EXTRACTED_V2 / f"{ticker}_business.txt").write_text(business_text_accum[:20000],encoding="utf-8")
        (SEC_10K_EXTRACTED_OLD / f"{ticker}_business.txt").write_text(business_text_accum[:20000],encoding="utf-8")
    if risk_text_accum:
        (SEC_10K_EXTRACTED_V2 / f"{ticker}_risk.txt").write_text(risk_text_accum[:20000],encoding="utf-8")
        (SEC_10K_EXTRACTED_OLD / f"{ticker}_risk.txt").write_text(risk_text_accum[:20000],encoding="utf-8")
    if mda_text_accum:
        (SEC_10K_EXTRACTED_V2 / f"{ticker}_mda.txt").write_text(mda_text_accum[:20000],encoding="utf-8")
        (SEC_10K_EXTRACTED_OLD / f"{ticker}_mda.txt").write_text(mda_text_accum[:20000],encoding="utf-8")
    try:
        del all_chunks, business_text_accum, risk_text_accum, mda_text_accum
        gc.collect()
    except: pass
    cnt=0
    try: cnt=len(json.loads(out_path.read_text()))
    except: pass
    status="ok" if cnt>0 else "no_chunks"
    return {"ticker":ticker,"cik":cik_pad,"status":status,"chunk_count":cnt,"has_business":(SEC_10K_EXTRACTED_V2 / f"{ticker}_business.txt").exists(),"has_risk":(SEC_10K_EXTRACTED_V2 / f"{ticker}_risk.txt").exists(),"has_mda":(SEC_10K_EXTRACTED_V2 / f"{ticker}_mda.txt").exists(),"chunks_file":str(out_path),"filings_fetched":fetched_html_status}

def main():
    parser=argparse.ArgumentParser(description="V2 HTML-aware SEC Ingestor")
    parser.add_argument("--limit",type=int,default=20)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--ticker",type=str,default=None)
    parser.add_argument("--tickers",type=str,default=None)
    parser.add_argument("--delay",type=float,default=0.3)
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--force-sub",action="store_true")
    parser.add_argument("--filing-types",type=str,default="10-K")
    parser.add_argument("--universe",type=str,default="")
    parser.add_argument("--manifest",type=str,default="")
    args=parser.parse_args()
    uni_path=Path(args.universe) if args.universe else find_universe_file()
    universe=load_universe(uni_path)
    print(f"Universe file {uni_path} loaded {len(universe)}")
    filing_types_requested=[ft.strip().upper().replace(" ","") for ft in args.filing_types.split(",") if ft.strip()]
    normalized=[]
    for ft in filing_types_requested:
        if ft in ["DEF14A","DEF-14A","DEF14A","DEF 14A"]: normalized.append("DEF14A")
        else: normalized.append(ft)
    filing_types_requested=normalized
    filing_limits={}
    for ft in filing_types_requested:
        if ft=="10-K": filing_limits["10-K"]=1
        elif ft=="DEF14A": filing_limits["DEF14A"]=1; filing_limits["DEF 14A"]=1
        elif ft=="10-Q": filing_limits["10-Q"]=2
        elif ft=="8-K": filing_limits["8-K"]=5
        elif ft=="S-1": filing_limits["S-1"]=1
        else: filing_limits[ft]=1
    print(f"Filing limits (rank-ordered): {filing_limits} (Do NOT ingest 3,4,5,13D/G per guide)")
    if args.ticker: universe=[u for u in universe if u["ticker"]==args.ticker.upper()]
    elif args.tickers:
        wanted=set(t.upper() for t in args.tickers.split(","))
        universe=[u for u in universe if u["ticker"] in wanted]
    else:
        if args.limit and args.limit>0 and args.limit < len(universe):
            big5=["AAPL","META","MSFT","NVDA","TSLA"]
            front=[]; rest=[]; big_set=set(big5)
            for u in universe:
                if u["ticker"] in big_set: front.append(u)
            for u in universe:
                if u["ticker"] not in big_set: rest.append(u)
            front_sorted=sorted(front,key=lambda x: big5.index(x["ticker"]) if x["ticker"] in big5 else 99)
            universe=front_sorted+rest
    if args.limit==0: subset=universe[args.start:]
    else: subset=universe[args.start: args.start+args.limit] if not args.ticker and not args.tickers else universe
    print(f"Processing {len(subset)} tickers from offset {args.start} limit {args.limit} tickers={args.ticker or args.tickers or 'all'}")
    manifest_path=Path(args.manifest) if args.manifest else DATA_DIR / "sec_v2_manifest.json"
    existing_manifest={}
    if manifest_path.exists():
        try:
            data=json.loads(manifest_path.read_text())
            if isinstance(data,dict) and "entries" in data:
                for e in data["entries"]: existing_manifest[e.get("ticker")]=e
        except: pass
    new_entries=[]; ok_cnt=0; fail_cnt=0; total_chunks=0
    for idx,entry in enumerate(subset):
        print(f"\n[{idx+1}/{len(subset)}] {entry['ticker']} {entry['cik']} {entry['company'][:40]}")
        try:
            res=process_ticker(entry,filing_limits,delay=args.delay,force=args.force,force_sub_fetch=args.force_sub)
            print(f"  -> {res['status']} chunks={res.get('chunk_count',0)} biz={res.get('has_business')} risk={res.get('has_risk')} mda={res.get('has_mda')}")
            new_entries.append(res)
            if res["status"]=="ok" or res.get("chunk_count",0)>0:
                ok_cnt+=1; total_chunks+=res.get("chunk_count",0)
            else: fail_cnt+=1
        except Exception as e:
            print(f"  !! Exception processing {entry['ticker']}: {e}",file=sys.stderr)
            import traceback; traceback.print_exc()
            fail_cnt+=1
            new_entries.append({"ticker":entry["ticker"],"status":f"exception:{e}","chunk_count":0})
        gc.collect()
        if (idx+1)%10==0:
            merged={**existing_manifest}
            for e in new_entries: merged[e["ticker"]]=e
            all_entries=list(merged.values())
            manifest_data={"generated_at":datetime.utcnow().isoformat()+"Z","universe_file":str(uni_path),"filing_limits":filing_limits,"slice":{"start":args.start,"limit":args.limit,"processed_this_run":idx+1,"total_universe":len(universe)},"totals":{"ok":ok_cnt,"fail":fail_cnt,"total_chunks":total_chunks},"entries":all_entries}
            try:
                manifest_path.write_text(json.dumps(manifest_data,indent=2))
                print(f"  Manifest interim written {manifest_path} ok={ok_cnt} fail={fail_cnt} chunks={total_chunks}")
            except Exception as e: print(f"Failed manifest write {e}")
    merged={**existing_manifest}
    for e in new_entries: merged[e["ticker"]]=e
    all_entries=list(merged.values())
    manifest_data={"generated_at":datetime.utcnow().isoformat()+"Z","universe_file":str(uni_path),"filing_limits":filing_limits,"slice":{"start":args.start,"limit":args.limit,"total_universe":len(universe),"processed_this_run":len(subset)},"totals":{"ok":ok_cnt,"fail":fail_cnt,"total_chunks":total_chunks,"total_entries":len(all_entries)},"entries":all_entries}
    manifest_path.write_text(json.dumps(manifest_data,indent=2))
    print(f"\nDONE V2 ingest manifest {manifest_path} ok={ok_cnt} fail={fail_cnt} chunks={total_chunks} total_entries={len(all_entries)}")
    chunks_index_path=CHUNKS_V2_DIR / "index.json"
    try:
        index_data=[]
        for jf in CHUNKS_V2_DIR.glob("*.json"):
            if jf.name=="index.json": continue
            try:
                ch=json.loads(jf.read_text())
                index_data.append({"ticker":jf.stem,"chunk_count":len(ch),"has_business":any(c.get("item_number")=="1" for c in ch),"has_risk":any(c.get("item_number")=="1A" for c in ch),"has_mda":any(c.get("item_number")=="7" for c in ch),"has_table":any(c.get("has_table") for c in ch)})
            except: continue
        index_data_sorted=sorted(index_data,key=lambda x:x["ticker"])
        chunks_index_path.write_text(json.dumps(index_data_sorted,indent=2))
        print(f"Chunks index written {chunks_index_path} with {len(index_data_sorted)} tickers")
    except Exception as e: print(f"Failed chunks index {e}")

if __name__=="__main__":
    main()
