import requests
import time
import pandas as pd
from typing import Dict, List

# Update base URL for your Jira tenant
BASE_URL = "https://cdssystems.atlassian.net/rest/tis/report/export"


def run_timepiece_report(api_key: str, report_type: str, jql: str, params: Dict = None, timeout_sec: int = 120) -> dict:
    """
    Run a Timepiece report (e.g., 'duration-by-status', 'transition-dates')
    and return JSON results.
    """
    headers = {"Authorization": f"tisjwt {api_key}"}
    body = {
        "reportType": report_type,
        "jql": jql,
        "format": "json"
    }
    if params:
        body.update(params)

    # 1. Initiate export
    start = requests.post(BASE_URL, headers=headers, json=body)
    print("DEBUG URL:", BASE_URL)
    print("DEBUG BODY:", body)
    print("DEBUG HEADERS:", headers)
    print("INIT RESPONSE:", start.status_code, start.text)

    start.raise_for_status()
    export_id = start.json()["id"]

    # 2. Poll until completed (with timeout)
    poll_url = f"{BASE_URL}/{export_id}"
    deadline = time.time() + timeout_sec
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"Report export did not complete within {timeout_sec} seconds")

        status_resp = requests.get(poll_url, headers=headers)
        print("POLL RESPONSE:", status_resp.status_code, status_resp.text)

        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get("status")

        if status == "COMPLETED":
            break
        elif status in ("FAILED", "ERROR"):
            raise RuntimeError(f"Report export failed: {status_data}")

        time.sleep(3)

    # 3. Download completed report
    download_url = f"{BASE_URL}/{export_id}/Download"
    download = requests.get(download_url, headers=headers)
    print("DOWNLOAD RESPONSE:", download.status_code)

    download.raise_for_status()

    try:
        return download.json()
    except Exception:
        return {"raw": download.text}


def parse_status_rules(rules_text: str) -> Dict[str, List[str]]:
    """
    Parse textarea rules like:
    Development = In Dev, Implementation
    Review = QA, Test
    On Hold = Blocked, Waiting
    """
    rules = {}
    for line in rules_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        bucket, statuses = line.split("=", 1)
        rules[bucket.strip()] = [s.strip() for s in statuses.split(",") if s.strip()]
    return rules


def build_dataframe(duration_data: dict, transition_data: dict, status_rules: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Merge Duration by Status + Transition Dates into one DataFrame.
    """
    rows = []

    # Lookup for transition dates
    transition_lookup = {}
    for issue in transition_data.get("issues", []):
        key = issue["key"]
        transitions = issue.get("transitions", [])
        start_date = next((t["date"] for t in transitions if t["to"] == "Development"), None)
        end_date = next((t["date"] for t in transitions if t["to"] == "Done"), None)
        transition_lookup[key] = {"Start": start_date, "End": end_date}

    # Process durations
    for issue in duration_data.get("issues", []):
        key = issue["key"]
        row = {"Key": key}
        total_days = 0

        for bucket, statuses in status_rules.items():
            dur = sum(
                (s["durationSeconds"] for s in issue.get("durations", []) if s["statusName"] in statuses),
                0
            )
            row[bucket] = dur / 86400.0
            total_days += dur / 86400.0

        row["CT"] = total_days
        row["Start"] = transition_lookup.get(key, {}).get("Start")
        row["End"] = transition_lookup.get(key, {}).get("End")
        rows.append(row)

    return pd.DataFrame(rows)
