"""Robust single-threaded fetcher for SEC submissions, avoids tmp collision and uses -u logging"""

import json
import os
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "sec_submissions"
CACHE.mkdir(parents=True, exist_ok=True)

USER_AGENT = "VectorEquities research (contact via GitHub)"


def curl_fetch(cik_pad):
    url = f"https://data.sec.gov/submissions/CIK{cik_pad}.json"
    tmp = f"/tmp/sec_fetch_{cik_pad}.json"
    # remove stale
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except:
        pass
    cmd = [
        "curl",
        "-sL",
        "--max-time",
        "60",
        "-H",
        f"User-Agent: {USER_AGENT}",
        "-H",
        "Accept: application/json",
        url,
        "-o",
        tmp,
        "-w",
        "%{http_code}",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=70)
        code = res.stdout.strip()[-3:]
        if os.path.exists(tmp) and os.path.getsize(tmp) > 5000:
            try:
                with open(tmp) as f:
                    data = json.load(f)
                return data, code
            except Exception as e:
                # try read raw
                print(
                    f"  parse fail cik {cik_pad} size {os.path.getsize(tmp)} err {e}",
                    flush=True,
                )
                return None, code
        else:
            # check why
            size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            print(
                f"  curl no file or small cik {cik_pad} code {code} size {size} stderr {res.stderr[:200]}",
                flush=True,
            )
            return None, code
    except Exception as e:
        print(f"  curl exception {cik_pad} {e}", flush=True)
        return None, "exc"


def fetch_sub(cik_pad, max_retries=4):
    out = CACHE / f"sub_{cik_pad}.json"
    if out.exists() and out.stat().st_size > 5000:
        try:
            # quick validate json
            data = json.loads(out.read_text()[:1000000])
            # ensure key cik exists
            if "cik" in data or "filings" in data:
                return data, True
        except:
            try:
                out.unlink()
            except:
                pass
    # need fetch
    for attempt in range(max_retries):
        data, code = curl_fetch(cik_pad)
        if data:
            try:
                out.write_text(json.dumps(data))
                # small sleep after success
                time.sleep(0.25 + random.random() * 0.15)
                return data, True
            except Exception as e:
                print(f" write fail {e}", flush=True)
                time.sleep(1)
        else:
            # retry logic: if 403 maybe wait longer, 429 etc
            sleep_t = 1 + attempt * 1.5 + random.random()
            if code in ("429", "403", "503"):
                sleep_t += 2
            print(
                f"  retry {attempt + 1}/{max_retries} cik {cik_pad} code {code} sleep {sleep_t:.1f}s",
                flush=True,
            )
            time.sleep(sleep_t)
    return None, False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()
    uni = json.loads((ROOT / "pipeline" / "data" / "universe.json").read_text())
    subset = uni[args.start : args.start + args.limit]
    print(f"Top {len(subset)} from {args.start} total universe {len(uni)}", flush=True)
    ok = 0
    fail = 0
    results = []
    for idx, entry in enumerate(subset):
        cik = str(entry["cik"]).zfill(10)
        ticker = entry["ticker"]
        data, success = fetch_sub(cik)
        if success and data:
            ok += 1
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            # count
            def14a = len([f for f in forms if "DEF 14A" in f or "DEF14A" in f])
            form4 = len([f for f in forms if f == "4"])
            print(
                f"[{idx + 1}/{len(subset)}] {ticker} {cik} OK DEF14A:{def14a} Form4:{form4} total:{len(forms)}  running ok {ok}",
                flush=True,
            )
            results.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "ok": True,
                    "def14a_recent": def14a,
                    "form4_recent": form4,
                }
            )
        else:
            fail += 1
            print(
                f"[{idx + 1}/{len(subset)}] {ticker} {cik} FAIL running fail {fail}",
                flush=True,
            )
            results.append({"cik": cik, "ticker": ticker, "ok": False})
            time.sleep(0.5)
        # periodic counting
        if (idx + 1) % 25 == 0:
            cnt = len(list(CACHE.glob("sub_*.json")))
            print(f"--- checkpoint {idx + 1} cache sub count {cnt} ---", flush=True)

    cnt_final = len(list(CACHE.glob("sub_*.json")))
    print(
        f"Done {ok}/{len(subset)} fail {fail} final cache sub count {cnt_final}",
        flush=True,
    )
