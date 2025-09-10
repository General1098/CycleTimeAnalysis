from __future__ import annotations
import io, time
from typing import Dict, Any
import pandas as pd
import requests
from urllib.parse import urlencode

def build_query_params(p: Dict[str, Any], for_export: bool=False, agg_for_export: str="average") -> Dict[str, Any]:
    q = {}
    q["aggregationType"] = p.get("aggregationType", "average")
    q["filterType"] = p.get("filterType", "jqlfilter")
    q["columnsBy"] = p.get("columnsBy", "statusDuration")
    if "page" in p: q["page"] = p["page"]
    if "pageSize" in p: q["pageSize"] = p["pageSize"]

    if q["filterType"] == "jqlfilter":
        if "jqlFilterID" in p: 
            q["jqlFilterID"] = p["jqlFilterID"]
        elif "filterId" in p: 
            q["jqlFilterID"] = p["filterId"]
    elif q["filterType"] == "project" and "projectKey" in p:
        q["projectKey"] = p["projectKey"]
    elif q["filterType"] == "customjql" and "jql" in p:
        q["jql"] = p["jql"]

    if for_export:
        mapping = {
            "average": "csvaverage",
            "sum": "csvsum",
            "median": "csvmedian",
            "standardDeviation": "csvstddev"
        }
        q["outputType"] = mapping.get(agg_for_export, "csvaverage")
    else:
        q["outputType"] = "json"
    return q

class TimepieceClient:
    def __init__(self, api_host: str, token: str, timeout: int = 60):
        self.host = api_host.rstrip("/")
        self.timeout = timeout
        self.token = (token or "").strip()
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"TISJWT {self.token}"})
        self.session.headers.update({"User-Agent": "CycleTimeTracker/1.7"})

    def aggregation_csv(self, params: Dict[str, Any]) -> bytes:
        q = build_query_params(params, for_export=False)
        q["outputType"] = "csv"
        url = f"{self.host}/rest/aggregation?{urlencode(q)}"
        r = self.session.post(url, timeout=self.timeout, headers={"Accept": "text/csv"})
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def aggregation_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        q = build_query_params(params, for_export=False)
        q["outputType"] = "json"
        url = f"{self.host}/rest/aggregation?{urlencode(q)}"
        r = self.session.post(url, timeout=self.timeout, headers={"Accept": "application/json"})
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def aggregation_json_df(self, params: Dict[str, Any]) -> pd.DataFrame:
        data = self.aggregation_json(params)
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            cols = [c.get("name") if isinstance(c, dict) else str(c) for c in data["columns"]]
            return pd.DataFrame(data["rows"], columns=cols)
        return pd.json_normalize(data)

    def export_start(self, params: Dict[str, Any], agg_choice: str) -> str:
        q = build_query_params(params, for_export=True, agg_for_export=agg_choice)
        url = f"{self.host}/rest/export?{urlencode(q)}"
        r = self.session.post(url, timeout=self.timeout, headers={"Accept": "application/json"})
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        data = r.json()
        return str(data.get("id") or data.get("exportId") or "")

    def export_status(self, export_id: str) -> Dict[str, Any]:
        url = f"{self.host}/rest/export/{export_id}"
        r = self.session.get(url, timeout=self.timeout, headers={"Accept": "application/json"})
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def export_download(self, export_id: str) -> bytes:
        url = f"{self.host}/rest/export/{export_id}/download"
        r = self.session.get(url, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def export_to_dataframe(self, params: Dict[str, Any], agg_choice: str) -> pd.DataFrame:
        export_id = self.export_start(params, agg_choice)
        for _ in range(30):
            time.sleep(2)
            info = self.export_status(export_id)
            if str(info.get("status") or info.get("state") or "").lower() in ("completed","done","finished","ready"):
                csv_bytes = self.export_download(export_id)
                return pd.read_csv(io.BytesIO(csv_bytes))
        raise RuntimeError("Export timed out")
