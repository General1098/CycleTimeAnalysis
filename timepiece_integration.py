import requests
import pandas as pd
from typing import Dict, List

BASE_URL = "https://tis.obss.io/rest/list2"


def fetch_status_durations(api_key: str, filter_id: str) -> dict:
    api_key = (api_key or "").strip()
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    def _call(fields: str):
        params = {
            "fields": fields,
            "filterType": "jqlfilter",
            "jqlFilterID": str(filter_id).strip(),
            "columnsBy": "statusduration",
            "outputType": "json",
            "calendar": "normalHours",
            "multiVisitBehavior": "total",
            "pageSize": 1000,
        }
        resp = requests.post(BASE_URL, headers=headers, data=params)
        if resp.status_code >= 400:
            # raise with body for easier debugging in the UI
            raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {resp.text}", response=resp)
        return resp.json()

    # Some Timepiece instances reject unknown fields; try Summary first, then retry without.
    try:
        return _call("summary,issuetype,customfield_10021")
    except requests.HTTPError as e:
        if "summary" in str(e).lower() and "400" in str(e):
            return _call("issuetype,customfield_10021")
        raise


def fetch_transition_dates(api_key: str, filter_id: str) -> dict:
    api_key = (api_key or "").strip()
    headers = {
        "Authorization": f"TISJWT {api_key}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    def _call(fields: str):
        params = {
            "fields": fields,
            "filterType": "jqlfilter",
            "jqlFilterID": str(filter_id).strip(),
            "columnsBy": "firstTransitionToStatusDate",
            "outputType": "json",
            "calendar": "normalHours",
            "pageSize": 1000,
        }
        resp = requests.post(BASE_URL, headers=headers, data=params)
        if resp.status_code >= 400:
            raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {resp.text}", response=resp)
        return resp.json()

    try:
        return _call("summary,issuetype,customfield_10021")
    except requests.HTTPError as e:
        if "summary" in str(e).lower() and "400" in str(e):
            return _call("issuetype,customfield_10021")
        raise


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



def _extract_field_value(val):
    """Return a clean string from OBSS fieldColumns 'value' which can be str | dict | list."""
    if isinstance(val, dict):
        return val.get("value")
    if isinstance(val, list):
        parts = []
        for v in val:
            if isinstance(v, dict):
                vv = v.get("value")
                if vv:
                    parts.append(str(vv))
            elif isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return ", ".join(parts) if parts else None
    if isinstance(val, str):
        return val.strip() or None
    return None


def build_dataframe(
    duration_data: dict,
    transition_data: dict,
    status_rules: Dict[str, List[str]],
    prefix_status_ids: Dict[str, Dict[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Build a dataframe of per-issue bucket durations + CT plus Start/End dates.

    `prefix_status_ids` allows overriding Start/Done status IDs per project prefix, e.g.:
    {
      "C7SM": {"start": "10111", "done": "10097"},
      "C7O":  {"start": "10206", "done": "10207"}
    }
    """
    rows = []
    status_lookup = build_status_lookup(duration_data, transition_data)
    rules_by_id = resolve_rules_to_ids(status_rules, duration_data)

    prefix_status_ids = prefix_status_ids or {}

    # IDs for start and end statuses across projects (fallback defaults)
    DEFAULT_START_IDS = ["10111", "10206", "10499"]  # SM, O, T4
    DEFAULT_END_IDS = ["10097", "10207", "10500"]    # SM, O, T4

    def _pick_ids_for_issue(issue_key: str):
        for pref, m in prefix_status_ids.items():
            if pref and issue_key.startswith(pref):
                s = m.get("start") or m.get("start_status_id")
                d = m.get("done") or m.get("done_status_id") or m.get("end")
                start_ids = [str(s)] if s else DEFAULT_START_IDS
                end_ids = [str(d)] if d else DEFAULT_END_IDS
                return start_ids, end_ids
        return DEFAULT_START_IDS, DEFAULT_END_IDS

    # --- Map transition dates by ID ---
    transition_lookup = {}
    for row in transition_data.get("table", {}).get("body", {}).get("rows", []):
        issue_key = next((c["value"] for c in row.get("headerColumns", []) if c["id"] == "issuekey"), None)
        if not issue_key:
            continue

        cols = {c["id"]: c.get("value") for c in row.get("valueColumns", [])}
        start_date = None
        end_date = None

        START_IDS, END_IDS = _pick_ids_for_issue(issue_key)

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

        # Summary can appear in either headerColumns (some Timepiece configs) or fieldColumns.
        header_map = {c.get("id"): c.get("value") for c in row.get("headerColumns", [])}

        # Parse Summary, IssueType and Sprint from fieldColumns
        field_map = {c.get('id'): c.get('value') for c in row.get('fieldColumns', [])}
        record['Summary'] = _extract_field_value(field_map.get('summary')) or header_map.get("summary")
        record['IssueType'] = _extract_field_value(field_map.get('issuetype'))
        record['Sprint'] = _extract_field_value(field_map.get('customfield_10021'))
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