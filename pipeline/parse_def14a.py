"""Heuristic DEF14A parser - extracts Summary Compensation Table without paid APIs"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DEF = ROOT / "pipeline" / "cache" / "sec_def14a"
CACHE_DEF.mkdir(parents=True, exist_ok=True)


# Simple HTML text extraction
def html_to_text(html_bytes):
    try:
        html = html_bytes.decode("utf-8", errors="ignore")
    except:
        html = str(html_bytes)
    # Remove scripts/styles
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Keep table structure
    html = re.sub(r"</tr>", "\nROW_END\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</td>|</th>", " | ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_comp_table_text(html_path):
    try:
        data = html_path.read_bytes()
        text = html_to_text(data)
        # Find Summary Compensation Table section
        # Look for "Summary Compensation Table" then capture next ~5000 chars
        idx = text.lower().find("summary compensation table")
        if idx == -1:
            # Try alternative
            idx = text.lower().find("summary compensation")
        if idx == -1:
            return None, text[:5000]
        snippet = text[idx : idx + 8000]
        return snippet, text
    except Exception as e:
        return None, f"error {e}"


def parse_neos_heuristic(html_path):
    """Very heuristic: find names near CEO/CFO and dollar amounts $"""
    try:
        raw = html_path.read_bytes().decode("utf-8", errors="ignore")
    except:
        return []
    # Find tables containing $ and Total and Name
    # Use regex to find rows with name pattern + dollar
    # Pattern: Name (capitalized words) near $ amount
    # Simpler: extract all <tr> that have $ and (CEO|Chief|President)
    neos = []
    # Find all tables
    tables = re.findall(r"<table.*?</table>", raw, flags=re.DOTALL | re.IGNORECASE)
    candidate_tables = []
    for tbl in tables:
        low = tbl.lower()
        if "summary compensation" in low or (
            "name" in low and "total" in low and "$" in tbl
        ):
            candidate_tables.append(tbl)
    # If found candidate tables, parse first 2
    for tbl in candidate_tables[:2]:
        # extract rows
        rows = re.findall(r"<tr.*?</tr>", tbl, flags=re.DOTALL | re.IGNORECASE)
        for row in rows[1:10]:  # skip header, take up to 9 rows
            # extract cells
            cells = re.findall(
                r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.DOTALL | re.IGNORECASE
            )
            if len(cells) < 3:
                continue
            # Clean cells
            clean_cells = []
            for c in cells:
                c = re.sub(r"<[^>]+>", " ", c)
                c = re.sub(r"\s+", " ", c).strip()
                clean_cells.append(c)
            # Heuristic: first cell looks like name (2+ capitalized words)
            first = clean_cells[0]
            if len(first) < 4 or len(first) > 80:
                continue
            # Must contain at least 2 words and not be pure numbers
            words = first.split()
            if len(words) < 2:
                continue
            if re.search(r"^\$|^\d", first):
                continue
            # Look for dollar in last cell
            last = clean_cells[-1]
            # Extract dollar amount
            m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", last)
            if not m and len(clean_cells) >= 4:
                # try second last
                m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", clean_cells[-2])
            if m:
                comp_str = m.group(1).replace(",", "")
                try:
                    comp = float(comp_str)
                    # Filter: comp likely > 100k and < 500M
                    if comp < 100000 or comp > 500000000:
                        # maybe in thousands? try *1000
                        if comp < 1000:
                            comp *= 1000
                    if comp >= 500000:  # threshold for NEO
                        neos.append(
                            {
                                "name": first[:60],
                                "title_guess": first,
                                "total_comp_raw": clean_cells[-1],
                                "total_comp": comp,
                                "cells": clean_cells[:4],
                            }
                        )
                except:
                    pass
    # Dedupe by name
    seen = set()
    uniq = []
    for n in neos:
        nn = n["name"].lower()
        if nn not in seen:
            seen.add(nn)
            uniq.append(n)
            if len(uniq) >= 6:
                break
    return uniq


def parse_def14a_file(html_path):
    ticker_date = html_path.stem  # e.g., MMM_2024-03-13_0000066740-24-000036
    parts = ticker_date.split("_")
    ticker = parts[0] if parts else "UNK"
    fdate = parts[1] if len(parts) > 1 else ""
    accession = parts[2] if len(parts) > 2 else ""
    snippet, full_text = extract_comp_table_text(html_path)
    neos = parse_neos_heuristic(html_path)
    # Board size heuristic: search "board of directors ... consists of X"
    board_size = None
    m = (
        re.search(r"board.*consists of (\d+)", full_text[:20000], re.IGNORECASE)
        if snippet
        else None
    )
    if m:
        try:
            board_size = int(m.group(1))
        except:
            pass
    if not board_size:
        m = re.search(r"(\d+)\s*directors", full_text[:10000], re.IGNORECASE)
        if m:
            try:
                board_size = int(m.group(1))
                if board_size > 20 or board_size < 3:
                    board_size = None
            except:
                pass
    result = {
        "ticker": ticker,
        "filing_date": fdate,
        "accession": accession,
        "file": str(html_path),
        "snippet_preview": (snippet[:1000] if snippet else "")[:1000],
        "neos": neos,
        "neo_count": len(neos),
        "board_size": board_size,
        "parse_success": len(neos) >= 2,
    }
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    # list html files
    html_files = sorted(CACHE_DEF.glob("*.html"))
    print(f"Found {len(html_files)} DEF14A html files")
    parsed = []
    for hf in html_files[: args.limit * 12]:  # approx
        res = parse_def14a_file(hf)
        parsed.append(res)
        print(
            f"{hf.name}: neos {res['neo_count']} success {res['parse_success']} board {res['board_size']} names {[n['name'][:20] for n in res['neos'][:3]]}"
        )
    # Save master
    out = ROOT / "pipeline" / "data" / "def14a_parsed.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in parsed:
            f.write(json.dumps(p) + "\n")
    print(
        f"Wrote {out} {len(parsed)} records, success rate {sum(1 for p in parsed if p['parse_success'])}/{len(parsed)}"
    )
