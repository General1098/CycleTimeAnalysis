import requests
import time
import pandas as pd
from typing import Dict, List

BASE_URL = "https://tis.obss.io"

def run_timepiece_report(api_key: str, report_type: str, jql: str, params: Dict = None) -> dict:
    """Run a Timepiece report and return JSON."""
    headers = {"Authorization": f"tisjwt {api_key}"}
    start = requests.post(
        f"{BASE_URL}/rest/report/{report_type}/export",
        headers=headers,
        params={"jql": jql, **(params or {})}
    )
    start.raise_for_status()
    export_id = start.json()["id"]

    while True:
        status = requests.get(f"{BASE_URL}/rest/report/export/{export_id}", headers=headers)
        status.raise_for_status()
        data = status.json()
        if data["status"] == "COMPLETED":
            break
        elif data["status"] in ("FAILED", "ERROR"):
            raise RuntimeError("Export failed")
        time.sleep(3)

    download = requests.get(f"{BASE_URL}/rest/report/export/{export_id}/download", headers=headers)
    download.raise_for_status()
    return download.json()


def parse_status_rules(rules_text: str) -> Dict[str, List[str]]:
    """Parse textarea rules like 'Development = In Dev, Implementation' into dict."""
    rules = {}
    for line in rules_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        bucket, statuses = line.split("=", 1)
        rules[bucket.strip()] = [s.strip() for s in statuses.split(",") if s.strip()]
    return rules


def build_dataframe(duration_data: dict, transition_data: dict, status_rules: Dict[str, List[str]]) -> pd.DataFrame:
    """Merge Duration by Status + Transition Dates into one DataFrame."""
    rows = []

    transition_lookup = {}
    for issue in transition_data.get("issues", []):
        key = issue["key"]
        transitions = issue.get("transitions", [])
        start_date = next((t["date"] for t in transitions if t["to"] == "Development"), None)
        end_date = next((t["date"] for t in transitions if t["to"] == "Done"), None)
        transition_lookup[key] = {"Start": start_date, "End": end_date}

    for issue in duration_data.get("issues", []):
        key = issue["key"]
        row = {"Key": key}
        total_days = 0

        for bucket, statuses in status_rules.items():
            dur = sum(
                (s["durationSeconds"] for s in issue["durations"] if s["statusName"] in statuses),
                0
            )
            row[bucket] = dur / 86400.0
            total_days += dur / 86400.0

        row["CT"] = total_days
        row["Start"] = transition_lookup.get(key, {}).get("Start")
        row["End"] = transition_lookup.get(key, {}).get("End")
        rows.append(row)

    return pd.DataFrame(rows)
