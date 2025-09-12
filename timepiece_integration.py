import requests
import pandas as pd
from typing import Dict, List

BASE_URL = "https://tis.obss.io/rest/list2"


def fetch_status_durations(api_key: str, filter_id: str) -> dict:
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    params = {
        "filterType": "jqlfilter",
        "jqlFilterID": filter_id,
        "columnsBy": "statusduration",
        "outputType": "json",
        "calendar": "normalHours",
        "multiVisitBehavior": "total",
        "pageSize": 1000,
    }
    resp = requests.post(BASE_URL, headers=headers, data=params)
    resp.raise_for_status()
    return resp.json()


def fetch_transition_dates(api_key: str, filter_id: str) -> dict:
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    params = {
        "filterType": "jqlfilter",
        "jqlFilterID": filter_id,
        "columnsBy": "firstTransitionToStatusDate",
        "outputType": "json",
        "calendar": "normalHours",
        "pageSize": 1000,
    }
    resp = requests.post(BASE_URL, headers=headers, data=params)
    resp.raise_for_status()
    return resp.json()


def build_status_lookup(*datasets: dict) -> Dict[str, str]:
    """
    Build a merged lookup of status ID -> Name from multiple API datasets.
    Ensures we capture *all* statuses from all projects.
    """
    lookup = {}

    for data in datasets:
        # 1. From includedStatuses
        for st in data.get("includedStatuses", []):
            sid = st.get("id")
            name = st.get("name")
            if sid and name:
                lookup[sid] = name

        # 2. From table header
        header = data.get("table", {}).get("header", {})
        for c in header.get("valueColumns", []):
            lookup[c["id"]] = c["value"]

        # 3. From table body rows
        for row in data.get("table", {}).get("body", {}).get("rows", []):
            for c in row.get("valueColumns", []):
                if c.get("id") and c.get("value"):
                    lookup[c["id"]] = c["value"]

    return lookup


def parse_status_rules(rules_text: str) -> Dict[str, List[str]]:
    rules = {}
    for line in rules_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        bucket, statuses = line.split("=", 1)
        rules[bucket.strip()] = [s.strip() for s in statuses.split(",") if s.strip()]
    return rules


def resolve_rules_to_ids(status_rules: Dict[str, List[str]], duration_data: dict) -> Dict[str, List[str]]:
    """
    Convert rules with status names into rules with status IDs,
    using the header of the duration_data response as the source of truth.
    """
    # Build name → id map from header
    name_to_id = {c["value"]: c["id"] for c in duration_data.get("table", {}).get("header", {}).get("valueColumns", [])}

    resolved = {}
    for bucket, names in status_rules.items():
        ids = []
        for name in names:
            if name in name_to_id:
                ids.append(name_to_id[name])
        resolved[bucket] = ids
    return resolved

def format_days(days: float) -> str:
    if pd.isna(days):
        return "-"
    total_minutes = int(days * 24 * 60)
    d, rem = divmod(total_minutes, 1440)
    h, m = divmod(rem, 60)
    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0:
        parts.append(f"{h}h")
    if d == 0 and m > 0:  # only show minutes for < 1 day
        parts.append(f"{m}m")
    return " ".join(parts) if parts else "0d"


def build_dataframe(duration_data: dict, transition_data: dict, status_rules: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    status_lookup = build_status_lookup(duration_data, transition_data)
    rules_by_id = resolve_rules_to_ids(status_rules, duration_data)

    # --- Map transition dates by ID ---
    transition_lookup = {}
    for row in transition_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next((c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"), None)
        if not issue_key:
            continue

        cols = {c["id"]: c.get("value") for c in row.get("valueColumns", [])}
        start_date = None
        end_date = None

        # IDs for start and end statuses across projects
        START_IDS = ["10111", "10206", "10499"]  # SM, O, T4
        END_IDS = ["10097", "10207", "10500"]    # SM, O, T4

        for sid in START_IDS:
            if sid in cols and cols[sid] not in (None, "-", ""):
                start_date = cols[sid]
                break

        for sid in END_IDS:
            if sid in cols and cols[sid] not in (None, "-", ""):
                end_date = cols[sid]
                break

        transition_lookup[issue_key] = {"Start": start_date, "End": end_date}

    # --- Durations by ID ---
    for row in duration_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next((c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"), None)
        if not issue_key:
            continue

        value_cols = {c["id"]: c.get("raw") for c in row.get("valueColumns", [])}
        record = {"Key": issue_key}
        total_days = 0

        for bucket, ids in rules_by_id.items():
            dur = 0
            for sid in ids:
                if sid in value_cols and value_cols[sid] not in (None, "-", ""):
                    try:
                        dur += float(value_cols[sid]) / 86400000.0  # ms → days
                    except Exception:
                        pass
            record[bucket] = dur
            total_days += dur

        record["CT"] = total_days
        record["Start"] = transition_lookup.get(issue_key, {}).get("Start")
        record["End"] = transition_lookup.get(issue_key, {}).get("End")
        rows.append(record)

    return pd.DataFrame(rows)
