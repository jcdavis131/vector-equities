"""SEC fetcher robust via curl fallback"""

import http.client
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache" / "sec"
CACHE.mkdir(parents=True, exist_ok=True)

USER_AGENT = "VectorEquities research (contact via GitHub)"


def curl_fetch_json(url, timeout=90):
    # use curl for robustness
    tmp = "/tmp/sec_fetch_tmp.json"
    try:
        cmd = [
            "curl",
            "-sL",
            "--max-time",
            str(timeout),
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
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        res.stdout.strip()[-3:] if res.stdout.strip() else ""
        # Actually curl -w outputs after, but we wrote file; check http code via separate?
        # We'll just check file exists and size
        if os.path.exists(tmp) and os.path.getsize(tmp) > 5000:
            try:
                with open(tmp) as f:
                    return json.load(f)
            except Exception:
                # try partial?
                try:
                    txt = open(tmp).read()
                    return json.loads(txt)
                except:
                    return None
    except Exception:
        pass
    return None


def robust_fetch_json(url, max_retries=5):
    for attempt in range(max_retries):
        # try urllib first quick
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
                return json.loads(data.decode("utf-8"))
        except http.client.IncompleteRead:
            # try curl
            j = curl_fetch_json(url)
            if j:
                return j
            time.sleep(1 + attempt)
            continue
        except Exception:
            # try curl as fallback
            j = curl_fetch_json(url)
            if j:
                return j
            time.sleep(1 + attempt * 1.2)
            if attempt == max_retries - 1:
                return None
    return None


def fetch_json(url):
    return robust_fetch_json(url)


def fetch_company_facts(cik: str):
    cik_pad = str(cik).zfill(10)
    cache_file = CACHE / f"facts_{cik_pad}.json"
    if cache_file.exists():
        try:
            if cache_file.stat().st_size > 1000:
                return json.loads(cache_file.read_text()[:20000000])
        except:
            try:
                cache_file.unlink()
            except:
                pass
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_pad}.json"
    data = robust_fetch_json(url)
    if data:
        try:
            cache_file.write_text(json.dumps(data))
        except:
            pass
    time.sleep(0.2)
    return data


def parse_financials_from_facts(facts):
    return {}
