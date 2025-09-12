import requests
import pandas as pd
from typing import Dict, List

BASE_URL = "https://tis.obss.io/rest/list2"


def fetch_status_durations(api_key: str, filter_id: str) -> dict:
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    params = {
        "filterType": "jqlfilter",
        "jqlFilterID": filter_id,
        "columnsBy": "statusduration",
        "outputType": "json",
        "calendar": "normalHours",
        "multiVisitBehavior": "total",
    }
    resp = requests.post(BASE_URL, headers=headers, data=params)
    resp.raise_for_status()
    return resp.json()


def fetch_transition_dates(api_key: str, filter_id: str) -> dict:
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    params = {
        "filterType": "jqlfilter",
        "jqlFilterID": filter_id,
        "columnsBy": "firstTransitionToStatusDate",
        "outputType": "json",
        "calendar": "normalHours",
    }
    resp = requests.post(BASE_URL, headers=headers, data=params)
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


def build_dataframe(
    duration_data: dict, transition_data: dict, status_rules: Dict[str, List[str]]
) -> pd.DataFrame:
    rows = []

    # --- Map transition dates (Start & End) ---
    transition_lookup = {}
    for row in transition_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next(
            (c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"),
            None,
        )
        cols = {c["id"]: c.get("value") for c in row.get("valueColumns", [])}

        start_date = (
            cols.get("In Development (C7SM)")
            or cols.get("In Progress (C7O)")
            or cols.get("In Development (C7T4)")
        )
        end_date = (
            cols.get("Done (C7SM)")
            or cols.get("Done (C7O)")
            or cols.get("Done (C7T4)")
        )

        transition_lookup[issue_key] = {"Start": start_date, "End": end_date}

    # --- Durations & Current Status ---
    for row in duration_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next(
            (c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"),
            None,
        )
        if not issue_key:
            continue

        # Collect duration values
        value_cols = {c["id"]: c.get("raw") for c in row.get("valueColumns", [])}

        # Determine the most recent "current status"
        current_status = None
        if "currentState" in row:
            try:
                status_id = row["currentState"][0].get("id")
                # Map ID back to human-readable name if possible
                for col in row.get("valueColumns", []):
                    if col.get("id") == status_id:
                        current_status = col.get("value", status_id)
                        break
                if not current_status:  # fallback if not mapped
                    current_status = status_id
            except Exception:
                current_status = None

        record = {"Key": issue_key, "CurrentStatus": current_status}
        total_days = 0

        # Apply bucket rules
        for bucket, statuses in status_rules.items():
            dur = 0
            for s in statuses:
                if s in value_cols and value_cols[s] not in (None, "-", ""):
                    try:
                        dur += float(value_cols[s]) / 86400000.0  # ms → days
                    except Exception:
                        pass
            record[bucket] = dur
            total_days += dur

        record["CT"] = total_days
        record["Start"] = transition_lookup.get(issue_key, {}).get("Start")
        record["End"] = transition_lookup.get(issue_key, {}).get("End")
        rows.append(record)

    return pd.DataFrame(rows)
