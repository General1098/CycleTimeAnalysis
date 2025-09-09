from __future__ import annotations
import json, time, urllib.parse, random, io
from typing import Dict, Any

import pandas as pd
import requests

class TimepieceClient:
    def __init__(self, api_host: str, token: str, auth_mode: str = "header", send_params: str = "Query string",
                 max_retries:int=6, base_delay:float=1.5, export_fallback: bool=True, timeout: int = 60):
        self.host = api_host.rstrip("/")
        self.timeout = timeout
        self.auth_mode = auth_mode  # "header" or "query"
        self.send_params = send_params  # "Query string" | "JSON body" | "Form-encoded"
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.export_fallback = export_fallback
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CycleTimeTracker/1.3 (+export-default)"})
        self.token = token.strip()
        if auth_mode == "header":
            self.session.headers.update({"Authorization": f"TISJWT {self.token}"})

    def _url(self, path: str, q: Dict[str, Any] | None = None) -> str:
        url = f"{self.host}{path}"
        qp = q or {}
        if self.auth_mode == "query":
            qp["token"] = self.token
        if qp:
            from urllib.parse import urlencode
            return url + ("&" if "?" in url else "?") + urlencode(qp, doseq=True)
        return url

    # ---- Export (async) helpers ----
    def export_start(self, body: Dict[str, Any]) -> str:
        r = self.session.post(self._url("/report/export"), json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        data = r.json()
        return str(data.get("id") or data.get("exportId") or "")

    def export_status(self, export_id: str) -> Dict[str, Any]:
        r = self.session.get(self._url(f"/report/export/{export_id}"), timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def export_download(self, export_id: str) -> bytes:
        r = self.session.get(self._url(f"/report/export/{export_id}/download"), timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def export_flow_fetch_csv(self, params: Dict[str, Any], st=None) -> bytes:
        body = {"params": params, "format": "csv"}
        if st is not None: st.info("Starting async export…")
        export_id = self.export_start(body)
        if st is not None: st.info(f"Export started: {export_id}. Polling…")
        # simple poll loop
        for i in range(30):
            time.sleep(2)
            info = self.export_status(export_id)
            status = str(info.get("status") or info.get("state") or "").lower()
            if status in ("completed", "done", "finished", "ready"):
                if st is not None: st.success("Export ready — downloading…")
                return self.export_download(export_id)
            if status in ("failed", "error"):
                raise RuntimeError(f"Export failed: {info}")
        raise RuntimeError("Export timed out")

    # ---- Live aggregation (advanced) with 429 handling ----
    def _post_once(self, path: str, params: Dict[str, Any], accept: str) -> requests.Response:
        if self.send_params == "JSON body":
            req_headers = {"Accept": accept, "Content-Type": "application/json"}
            req_body = json.dumps(params)
            return self.session.post(self._url(path), data=req_body, timeout=self.timeout, headers=req_headers)
        elif self.send_params == "Form-encoded":
            from urllib.parse import urlencode
            req_headers = {"Accept": accept, "Content-Type": "application/x-www-form-urlencoded"}
            req_body = urlencode(params, doseq=True)
            return self.session.post(self._url(path), data=req_body, timeout=self.timeout, headers=req_headers)
        else:  # Query string
            req_headers = {"Accept": accept, "Content-Type": "application/json"}
            return self.session.post(self._url(path, q=params), timeout=self.timeout, headers=req_headers)

    def _post_with_retry_429(self, path: str, params: Dict[str, Any], accept: str, st = None) -> requests.Response:
        attempt = 0
        while True:
            r = self._post_once(path, params, accept)
            if r.status_code != 429:
                return r
            attempt += 1
            if attempt > self.max_retries:
                return r
            delay = self.base_delay * (2 ** (attempt - 1))
            delay = delay * (0.7 + 0.6 * random.random())
            if st is not None:
                st.info(f"429 rate-limit — waiting {delay:.1f}s before retry {attempt}/{self.max_retries}…")
            time.sleep(delay)

    def fetch_csv_with_robust_429(self, params: Dict[str, Any], st=None) -> bytes:
        r = self._post_with_retry_429("/rest/aggregation", params, "text/csv", st=st)
        if r.ok:
            return r.content
        raise RuntimeError(f"{r.status_code}: {r.text}")

    def fetch_json_with_robust_429(self, params: Dict[str, Any], st=None) -> pd.DataFrame:
        r = self._post_with_retry_429("/rest/aggregation", params, "application/json", st=st)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "rows" in data and "columns" in data:
                cols = [c.get("name") if isinstance(c, dict) else str(c) for c in data["columns"]]
                return pd.DataFrame(data["rows"], columns=cols)
            return pd.json_normalize(data)
        raise RuntimeError(f"{r.status_code}: {r.text}")
