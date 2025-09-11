import requests
import pandas as pd
from typing import Dict, List

BASE_URL = "https://tis.obss.io/rest/list2"

# ------------------------------
# Generic fetch with paging
# ------------------------------

def fetch_with_paging(api_key: str, params: dict) -> dict:
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    combined = None
    next_token = None

    while True:
        if next_token:
            params["nextPageToken"] = next_token
        elif "nextPageToken" in params:
            params.pop("nextPageToken")

        resp = requests.post(BASE_URL, headers=headers, data=params)
        resp.raise_for_status()
        data = resp.json()

        # Merge into combined result
        if combined is None:
            combined = data
        else:
            combined["table"]["body"]["rows"].extend(data["table"]["body"]["rows"])

        next_token = data.get("nextPageToken")
        if not next_token:
            break

    return combined


# ------------------------------
# Fetch reports from Timepiece
# ------------------------------

def fetch_status_durations(api_key: str, filter_id: str) -> dict:
    params = {
        "filterType": "jqlfilter",
        "jqlFilterID": filter_id,
        "columnsBy": "statusduration",
        "outputType": "json",
        "calendar": "normalHours",
        "multiVisitBehavior": "total",
        "pageSize": 100
    }
    return fetch_with_paging(api_key, params)


def fetch_transition_dates(api_key: str, filter_id: str) -> dict:
    params = {
        "filterType": "jqlfilter",
        "jqlFilterID": filter_id,
        "columnsBy": "firstTransitionToStatusDate",
        "outputType": "json",
        "calendar": "normalHours",
        "pageSize": 100
    }
    return fetch_with_paging(api_key, params)


# ------------------------------
# Helpers
# ------------------------------

def parse_status_rules(rules_text: str) -> Dict[str, List[str]]:
    rules = {}
    for line in rules_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        bucket, statuses = line.split("=", 1)
        rules[bucket.strip()] = [s.strip() for s in statuses.split(",") if s.strip()]
    return rules


# ------------------------------
# Build DataFrame from responses
# ------------------------------

def build_dataframe(duration_data: dict, transition_data: dict, status_rules: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []

    # ---- Build lookup maps ----
    duration_id_to_name = {
        c["id"]: c["value"]
        for c in duration_data.get("table", {}).get("header", {}).get("valueColumns", [])
    }
    transition_id_to_name = {
        c["id"]: c["value"]
        for c in transition_data.get("table", {}).get("header", {}).get("valueColumns", [])
    }

    # ---- Map transition dates ----
    transition_lookup = {}
    for row in transition_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next(
            (c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"), None
        )
        if not issue_key:
            continue

        cols = {
            transition_id_to_name.get(c["id"], c["id"]): c.get("value")
            for c in row.get("valueColumns", [])
        }

        # Start = first entry into any development-like status
        start_date = (
            cols.get("In Development (C7SM)")
            or cols.get("In Progress (C7O)")
            or cols.get("In Development (C7T4)")
        )
        
        # End = first entry into any Done/Complete-like status
        end_date = (
            cols.get("Done (C7SM)")
            or cols.get("Done (C7O)")
            or cols.get("Done (C7T4)")
            or cols.get("Complete (C7SM)")
            or cols.get("Complete (C7O)")
            or cols.get("Complete (C7T4)")
        )

        transition_lookup[issue_key] = {"Start": start_date, "End": end_date}

    # ---- Durations ----
    for row in duration_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next(
            (c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"), None
        )
        if not issue_key:
            continue

        # Convert IDs → names for value columns
        value_cols = {
            duration_id_to_name.get(c["id"], c["id"]): c.get("raw")
            for c in row.get("valueColumns", [])
        }

        record = {"Key": issue_key}
        total_days = 0

        for bucket, statuses in status_rules.items():
            dur = 0
            for s in statuses:
                if s in value_cols and value_cols[s] not in (None, "-", ""):
                    try:
                        dur += float(value_cols[s]) / 86400000.0  # raw is ms
                    except Exception:
                        pass
            record[bucket] = dur
            total_days += dur

        record["CT"] = total_days
        record["Start"] = transition_lookup.get(issue_key, {}).get("Start")
        record["End"] = transition_lookup.get(issue_key, {}).get("End")
        rows.append(record)

    return pd.DataFrame(rows)
