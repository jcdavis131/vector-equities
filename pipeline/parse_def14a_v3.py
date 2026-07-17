#!/usr/bin/env python3
"""
Fast DEF14A Parser v3 - optimized for memory/speed avoiding full-doc BeautifulSoup

- Extract tables via regex, only parse candidate snippets with BS html.parser
- Board size via regex
- XBRL via regex (existing)
- Same filtering as before but role filter >=2 words
- Streaming resume support
"""
import re, json, html, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DEF = ROOT / "pipeline" / "cache" / "sec_def14a"
OUT_JSONL = ROOT / "pipeline" / "data" / "def14a_parsed_v3.jsonl"

try:
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    HAS_BS4 = True
except:
    try:
        from bs4 import BeautifulSoup
        HAS_BS4 = True
    except:
        HAS_BS4 = False

BLACKLIST_STRICT = ["board", "committee", "compensation committee", "director compensation", "annual meeting", "audit committee", "independent director", "cash retainer"]
ROLE_KW = ["ceo", "cfo", "cto", "coo", "president", "chief", "executive officer", "general counsel", "chairman", "founder", "vice president", "evp", "svp"]
ROLE_TOKENS = ["chief", "president", "chairman", "executive", "officer", "vice president", "financial", "operating", "technology", "general counsel", "secretary", "treasurer", "founder", "senior vice", "senior", "vice", "compensation", "adjustment", "interim", "former", "paid", "chair", "director"]

TITLE_RE = re.compile(r"^[A-Z][a-zA-Z'\-\.]*"
                       r"(?:\s+[A-Z]\.?)?"
                       r"(?:\s+[A-Z][a-zA-Z'\-]*"
                       r"(?:\s+[A-Z][a-zA-Z'\-]*)?)?"
                       r"(?:,\s*(?:Jr\.?|Sr\.?|II|III|IV))?\s*$")
ALLCAPS_RE = re.compile(r"^[A-Z]{2,}(?:\s+[A-Z]\.?)?(?:\s+[A-Z]{2,}){1,3}(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?$")
SUFFIX_RE = re.compile(r",?\s*(Jr\.?|Sr\.?|II|III|IV)\.?$", re.I)

def clean_text(s):
    if not s:
        return ""
    s = html.unescape(s)
    s = s.replace("\xa0"," ").replace("\u200b"," ").replace("\u200c","").replace("\u200d","").replace("\u2060"," ").replace("\ufeff"," ")
    s = re.sub(r"\s+"," ",s).strip()
    return s

def clean_name(raw):
    if not raw:
        return ""
    s = clean_text(raw)
    s = re.sub(r"\s*\(\d+\)", "", s)
    s = re.sub(r"\s*\(\d+[a-z]?\)$", "", s)
    s = re.sub(r"\s*\[[^\]]+\]\s*$", "", s)
    s = re.sub(r"[\*†‡]+$","",s)
    s = s.strip(".,;: \t")
    parts = s.split()
    for idx in range(2, len(parts)):
        rest = " ".join(parts[idx:]).lower()
        if any(r in rest for r in ["chairman","chief","president","officer","executive","founder","director"]):
            cand = " ".join(parts[:idx])
            if 2 <= len(cand.split()) <=4:
                s = cand
                break
    if " - " in s:
        left = s.split(" - ")[0]
        if 2 <= len(left.split()) <=4:
            s = left
    return s.strip()

def is_blacklisted(name):
    low = name.lower()
    for b in BLACKLIST_STRICT:
        if b in low:
            return True
    # if contains board etc as whole token
    # reject if any blacklist token as separate word? Use simple contains for now for director compensation etc already strict
    # broader but only for longish phrase? We'll keep minimal to avoid false positives
    return False

def is_plausible_name(name):
    if not name:
        return False
    if len(name) < 6 or len(name) > 80:
        return False
    words = name.split()
    if len(words) <2 or len(words) >5:
        return False
    if re.search(r"[\d\$%]", name):
        return False
    low = name.lower()
    # quick blacklist
    if is_blacklisted(name):
        return False
    if low in ["name and principal position", "named executive officers", "principal position", "name"]:
        return False
    # role token check: reject if contains chief etc (as name shouldn't)
    if any(k in low for k in ROLE_TOKENS):
        return False
    # strict blacklist list for filler
    if any(bl in low for bl in ["summary compensation", "stock award", "option award", "all other compensation", "non-equity", "change in pension", "fiscal year", "year ended", "cash bonus"]):
        return False
    # regex
    if TITLE_RE.match(name):
        caps = sum(1 for w in words if w and w[0].isupper())
        if caps >=2:
            return True
    if ALLCAPS_RE.match(name) and len(name) <=45:
        if all(len(w)>=2 for w in words):
            return True
    if re.match(r"^[A-Z]\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+", name):
        return True
    if re.match(r"^[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+", name):
        return True
    return False

def extract_comp(cells):
    for txt in reversed(cells):
        if not txt:
            continue
        ct = clean_text(txt)
        matches = re.findall(r"\$?\s*([\d,]{4,12}(?:\.\d{1,2})?)", ct)
        for m in reversed(matches):
            try:
                v = float(m.replace(",",""))
                if 500_000 <= v < 300_000_000:
                    return v, ct
            except:
                continue
    return None, None

def score_table_text(tbl_text):
    low = tbl_text.lower()
    score = 0
    if "summary compensation table" in low:
        score += 100
    if "summary compensation" in low and "table" in low:
        score += 50
    if "salary" in low and "bonus" in low and "total" in low:
        score += 40
    if "stock awards" in low or "stock award" in low:
        score += 10
    if "option awards" in low:
        score += 10
    if "non-equity" in low:
        score += 10
    if "$" in tbl_text:
        score += 5
    years = re.findall(r"20[1-5]\d", tbl_text)
    if len(years) >=3:
        score += 10
    if "principal position" in low:
        score += 20
    if len(tbl_text) < 200:
        score -= 50
    if len(tbl_text) > 60000:
        score -= 10
    # Penalize footnote-style tables that reference Summary Compensation Table but lack salary column
    if "summary compensation table" in low and "salary" not in low:
        score -= 80
    if low.strip().startswith("(") and len(tbl_text) < 800:
        score -= 30
    return score

def strip_tags(s):
    return re.sub(r"<[^>]+>", " ", s)

def parse_table_snippet(tbl_html):
    """
    Parse a single table html snippet with BS, return list of NEOs
    """
    if not HAS_BS4:
        return []
    try:
        # Use html.parser for low memory
        soup = BeautifulSoup(tbl_html, "html.parser")
        rows = soup.find_all("tr")
        neos=[]
        if len(rows) <2:
            return []
        for r in rows[1:30]:
            cells = r.find_all(["td","th"])
            if len(cells) <2:
                continue
            texts = [clean_text(c.get_text(separator=" ", strip=True)) for c in cells]
            if len([t for t in texts if t]) <2:
                continue
            first_raw = texts[0] if texts else ""
            if not first_raw:
                continue
            if re.match(r"^\s*20[12]\d\s*$", first_raw):
                continue
            if re.match(r"^[\$\d,\.\s]+$", first_raw):
                continue
            if first_raw.lower() in ["name and principal position", "name", "total", ""]:
                continue
            name_candidate = clean_name(first_raw)
            comp_val = None
            comp_raw_txt = None
            role = None
            if not is_plausible_name(name_candidate):
                # try second col as name if first is year/blank
                if len(texts)>=2:
                    second_raw = texts[1]
                    second_clean = clean_name(second_raw)
                    if is_plausible_name(second_clean) and (re.match(r"^\d{4}$", first_raw) or first_raw=="" or first_raw in ["—","-"]):
                        name_candidate = second_clean
                        comp_val, comp_raw_txt = extract_comp(texts[2:])
                    else:
                        continue
                else:
                    continue
            else:
                comp_val, comp_raw_txt = extract_comp(texts[1:])
                if comp_val is None:
                    continue
            # already filtered role tokens in is_plausible_name, so name should be clean
            # role extraction from second col if present
            if len(texts) >=2 and texts[1]:
                # second col maybe role or year
                sec = clean_text(texts[1])
                low_sec = sec.lower()
                if any(k in low_sec for k in ROLE_KW) and not is_plausible_name(sec):
                    role = sec[:150]
            neos.append({"name": name_candidate, "role": role, "total_comp": comp_val, "comp_raw": comp_raw_txt})
        # dedupe
        dedup={}
        for n in neos:
            key=n["name"].lower().strip()
            if key not in dedup:
                dedup[key]=n
            else:
                if n["total_comp"] and (dedup[key]["total_comp"] is None or n["total_comp"]>dedup[key]["total_comp"]):
                    dedup[key]=n
        uniq=list(dedup.values())
        uniq_sorted=sorted(uniq, key=lambda x: x["total_comp"] or 0, reverse=True)
        return uniq_sorted[:10]
    except Exception as e:
        return []
    finally:
        # free memory
        try:
            soup.decompose()
        except:
            pass

def parse_xbrl_names(raw_html):
    found=[]
    try:
        pat = re.compile(r'<(?:ix:nonNumeric|xbrli:nonNumeric|span)[^>]*name=["\'](?:[^"\']*?)(?:PeoName|NameOfExecutiveOfficer|ExecutiveOfficerName|NameOf.*Officer)[^"\']*["\'][^>]*>([^<]{3,80})</[^>]+>', re.I)
        for m in pat.finditer(raw_html):
            raw_name=m.group(1)
            cn=clean_name(raw_name)
            if is_plausible_name(cn):
                found.append(cn)
        pat2 = re.compile(r'dimension=["\']ecd:ExecutiveNameAxis["\'][^>]*>.*?<[^>]*\.domain[^>]*>([^<]{3,80})</[^>]*>', re.I|re.DOTALL)
        for m in pat2.finditer(raw_html):
            raw_name=re.sub(r"<[^>]+>","",m.group(1))
            cn=clean_name(raw_name)
            if is_plausible_name(cn):
                found.append(cn)
        pat3 = re.compile(r'<ecd:ExecutiveNameAxis\.domain[^>]*>([^<]{3,80})</ecd:ExecutiveNameAxis\.domain>', re.I)
        for m in pat3.finditer(raw_html):
            cn=clean_name(m.group(1))
            if is_plausible_name(cn):
                found.append(cn)
        seen=set()
        uniq=[]
        for n in found:
            k=n.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(n)
        return uniq[:10]
    except:
        return []

def parse_one_file_fast(html_path: Path):
    try:
        raw = html_path.read_bytes().decode("utf-8", errors="ignore")
    except Exception as e:
        return {"ticker": html_path.name.split("_")[0], "filing_date": html_path.name.split("_")[1] if "_" in html_path.name else "", "file": f"pipeline/cache/sec_def14a/{html_path.name}", "neo_count":0, "neos":[], "board_size":None, "candidates_found":0, "parse_success": False, "method": "read-fail", "error": str(e)}
    ticker = html_path.name.split("_")[0]
    try:
        fdate = html_path.name.split("_")[1]
    except:
        fdate = ""
    board_size=None
    # board via regex (fast)
    m = re.search(r'board.*consists of (\d+)', raw, re.I)
    if not m:
        m = re.search(r'board of directors consists of (\d+)', raw, re.I)
    if m:
        try:
            bs=int(m.group(1))
            if 3 <= bs <=25:
                board_size=bs
        except:
            pass

    # Find tables via regex (avoid full soup)
    tables_raw = re.findall(r'<table[^>]*>.*?</table>', raw, flags=re.DOTALL|re.IGNORECASE)
    scored=[]
    for tbl_html in tables_raw:
        # quick pre-filter: must contain salary and bonus and total or summary compensation
        low = tbl_html.lower()
        if len(tbl_html) < 500:
            continue
        # quick text stripping for scoring (avoid BS)
        txt = strip_tags(tbl_html)
        txt = clean_text(txt)
        if len(txt) < 100:
            continue
        s = score_table_text(txt)
        if s > 20:
            scored.append((s, tbl_html, txt))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = len(scored)
    neos_final=[]
    method="v3-table-fast"

    # Try top scored tables with BS snippet parsing - increased to 25 to handle footnote table high scores
    for score, tbl_html, txt in scored[:25]:
        parsed = parse_table_snippet(tbl_html)
        if len(parsed) >=2:
            neos_final = parsed
            method = f"v3-table-score{score}"
            break

    # Anchor fallback: search for summary compensation table then next table
    if len(neos_final) <2:
        # find position of phrase
        low_raw = raw.lower()
        idx = low_raw.find("summary compensation table")
        if idx != -1:
            # find next <table after idx
            next_tbl_start = raw.lower().find("<table", idx)
            if next_tbl_start != -1:
                # find end of this table
                end = raw.lower().find("</table>", next_tbl_start)
                if end != -1:
                    snippet = raw[next_tbl_start:end+8]
                    parsed = parse_table_snippet(snippet)
                    if len(parsed) >=2:
                        neos_final = parsed
                        method = "v3-anchor-next-table"

    # XBRL fallback
    if len(neos_final) <2:
        xbrl_names = parse_xbrl_names(raw)
        if len(xbrl_names) >=2:
            # try lower scored tables - extended range
            for score, tbl_html, txt in scored[25:60]:
                parsed = parse_table_snippet(tbl_html)
                if len(parsed) >=2:
                    neos_final = parsed
                    method = f"v3-xbrl+table-{score}"
                    break
            if len(neos_final) <2:
                neos_final = [{"name": n, "role": None, "total_comp": None, "comp_raw": None} for n in xbrl_names[:8]]
                method = "v3-xbrl-names-only"

    # Final fallback line scan (light)
    if len(neos_final) <2:
        # Use raw lines
        lines = raw.splitlines()[:5000]
        temp=[]
        for line in lines:
            cl = clean_text(strip_tags(line))
            if len(cl) <10 or len(cl)>300:
                continue
            if "$" not in cl:
                continue
            words = cl.split()
            if len(words) <4:
                continue
            for wcnt in [3,2,4]:
                if len(words) < wcnt+1:
                    continue
                cand = " ".join(words[:wcnt])
                cand_clean = clean_name(cand)
                if is_plausible_name(cand_clean):
                    rest = " ".join(words[wcnt:])
                    comp_val, _ = extract_comp([rest])
                    if comp_val and comp_val>500_000:
                        temp.append({"name": cand_clean, "role": None, "total_comp": comp_val, "comp_raw": rest[:200]})
                        break
        if len(temp)>=2:
            dedup={}
            for n in temp:
                k=n["name"].lower()
                if k not in dedup or (n["total_comp"] and n["total_comp"]>(dedup[k]["total_comp"] or 0)):
                    dedup[k]=n
            uniq=list(dedup.values())
            uniq_sorted=sorted(uniq, key=lambda x: x["total_comp"] or 0, reverse=True)[:10]
            if len(uniq_sorted)>=2:
                neos_final=uniq_sorted
                method="v3-line-scan"

    # Final normalization title case for all caps
    final=[]
    seen=set()
    for n in neos_final:
        name=n.get("name","").strip()
        if not name:
            continue
        if name.isupper():
            name=" ".join([w.capitalize() for w in name.split()])
        name=clean_name(name)
        if not name:
            continue
        k=name.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append({"name": name, "role": n.get("role"), "total_comp": n.get("total_comp"), "comp_raw": n.get("comp_raw")})

    with_comp=[x for x in final if x["total_comp"] and x["total_comp"]>300_000]
    if len(with_comp)>=2:
        final=with_comp

    neo_count=len(final)
    parse_success=neo_count>=2
    return {
        "ticker": ticker,
        "filing_date": fdate,
        "file": f"pipeline/cache/sec_def14a/{html_path.name}",
        "neo_count": neo_count,
        "neos": final[:6],
        "board_size": board_size,
        "candidates_found": candidates,
        "parse_success": parse_success,
        "method": method
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cache = CACHE_DEF
    files = sorted(cache.glob("*.html"))
    print(f"Found {len(files)} html files in {cache}", flush=True)
    if args.limit:
        files = files[:args.limit]
        print(f"Limited to {len(files)}", flush=True)
    out = OUT_JSONL
    out.parent.mkdir(parents=True, exist_ok=True)

    already=set()
    if args.resume and out.exists():
        try:
            with open(out, "r") as f:
                for line in f:
                    try:
                        j=json.loads(line)
                        already.add(j.get("file",""))
                        already.add(Path(j.get("file","")).name)
                        already.add(j.get("file","").split("/")[-1])
                    except:
                        pass
            print(f"Resume: already {len(already)}", flush=True)
            orig_len=len(files)
            files=[fl for fl in files if fl.name not in already and f"pipeline/cache/sec_def14a/{fl.name}" not in already]
            print(f"Remaining {len(files)} of {orig_len}", flush=True)
        except Exception as e:
            print(f"Resume read fail {e}", flush=True)

    mode="a" if args.resume else "w"
    out_f=open(out, mode)
    total=0
    succ=0
    for i, f in enumerate(files, 1):
        try:
            r=parse_one_file_fast(f)
        except Exception as e:
            r={"ticker": f.name.split("_")[0], "filing_date": f.name.split("_")[1] if "_" in f.name else "", "file": f"pipeline/cache/sec_def14a/{f.name}", "neo_count":0, "neos":[], "board_size":None, "candidates_found":0, "parse_success": False, "method": f"crash {e}"}
        out_f.write(json.dumps(r)+"\n")
        out_f.flush()
        total+=1
        if r["parse_success"]:
            succ+=1
        if i%25==0 or r["parse_success"]:
            names=[n["name"] for n in r["neos"][:3]]
            print(f"{i}/{len(files)} {'OK' if r['parse_success'] else 'FAIL'} {f.name} neos {r['neo_count']} cand {r['candidates_found']} {r['method']} {names}", flush=True)
        if i%100==0:
            gc.collect()
    out_f.close()
    # final stats
    try:
        with open(out, "r") as fin:
            all_lines=[json.loads(l) for l in fin if l.strip()]
        succ=sum(1 for r in all_lines if r["parse_success"])
        total=len(all_lines)
        print(f"\nWrote {out} total={total} succ={succ} rate={succ/total:.3f}", flush=True)
        from collections import Counter
        print("Methods:", Counter([r["method"] for r in all_lines if r["parse_success"]]).most_common(15), flush=True)
    except Exception as e:
        print(f"stats fail {e}", flush=True)
