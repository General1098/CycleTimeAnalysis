import requests
import pandas as pd
from typing import Dict, List

def fetch_status_durations(api_key: str, base_url: str, jql: str) -> dict:
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    params = {
        "filterType": "customjql",
        "customjql": jql,
        "calendar": "normalHours",
        "columnsBy": "statusDuration",
        "outputType": "json"
    }
    resp = requests.post(base_url, headers=headers, data=params)
    resp.raise_for_status()
    return resp.json()

def fetch_transition_dates(api_key: str, base_url: str, jql: str) -> dict:
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    params = {
        "filterType": "customjql",
        "customjql": jql,
        "calendar": "normalHours",
        "columnsBy": "firstTransitionToStatusDate",
        "outputType": "json"
    }
    resp = requests.post(base_url, headers=headers, data=params)
    resp.raise_for_status()
    return resp.json()

def parse_status_rules(rules_text: str) -> Dict[str, List[str]]:
    rules = {}
    for line in rules_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        bucket, statuses = line.split("=", 1)
        rules[bucket.strip()] = [s.strip() for s in statuses.split(",") if s.strip()]
    return rules

def build_dataframe(duration_data: dict, transition_data: dict, status_rules: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    # Map transition dates
    transition_lookup = {}
    for row in transition_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next((c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"), None)
        cols = {c["id"]: c.get("value") for c in row.get("valueColumns", [])}
        start_date = cols.get("Development") or cols.get("In Development")
        end_date = cols.get("Done")
        transition_lookup[issue_key] = {"Start": start_date, "End": end_date}
    # Durations
    for row in duration_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next((c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"), None)
        if not issue_key:
            continue
        value_cols = {c["id"]: c.get("raw") for c in row.get("valueColumns", [])}
        record = {"Key": issue_key}
        total_days = 0
        for bucket, statuses in status_rules.items():
            dur = 0
            for s in statuses:
                if s in value_cols and value_cols[s] not in (None, "-", ""):
                    try:
                        dur += float(value_cols[s]) / 86400.0
                    except:
                        pass
            record[bucket] = dur
            total_days += dur
        record["CT"] = total_days
        record["Start"] = transition_lookup.get(issue_key, {}).get("Start")
        record["End"] = transition_lookup.get(issue_key, {}).get("End")
        rows.append(record)
    return pd.DataFrame(rows)
