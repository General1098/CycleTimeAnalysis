from __future__ import annotations
import json, time, urllib.parse, shlex
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import pandas as pd
import requests

@dataclass
class AttemptLog:
    auth_mode: str
    send_params: str
    method: str
    url: str
    status: int
    ok: bool
    error: str
    req_headers: Dict[str, str]
    req_body: str
    resp_headers: Dict[str, str]
    resp_body_excerpt: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "auth": self.auth_mode,
            "send": self.send_params,
            "status": self.status,
            "ok": self.ok,
            "error": (self.error[:140] if self.error else ""),
            "url": self.url[:160],
            "resp_excerpt": self.resp_body_excerpt[:160],
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TimepieceClient:
    def __init__(self, api_host: str, token: str, auth_mode: str = "header", send_params: str = "JSON body", timeout: int = 60):
        self.host = api_host.rstrip("/")
        self.timeout = timeout
        self.auth_mode = auth_mode  # "header" or "query"
        self.send_params = send_params  # "JSON body" | "Query string" | "Form-encoded"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CycleTimeTracker/1.0 (+diagnostics)",
        })
        self.token = token.strip()
        if auth_mode == "header":
            self.session.headers.update({"Authorization": f"TISJWT {self.token}"})

    # ---- Low-level helpers ----
    def _url(self, path: str, q: Dict[str, Any] | None = None) -> str:
        url = f"{self.host}{path}"
        qp = q or {}
        if self.auth_mode == "query":
            qp["token"] = self.token
        if qp:
            return url + ("&" if "?" in url else "?") + urllib.parse.urlencode(qp, doseq=True)
        return url

    def _post(self, path: str, params: Dict[str, Any], accept: str) -> Tuple[requests.Response, AttemptLog]:
        if self.send_params == "JSON body":
            req_headers = {"Accept": accept, "Content-Type": "application/json"}
            req_body = json.dumps(params)
            r = self.session.post(self._url(path), data=req_body, timeout=self.timeout, headers=req_headers)
        elif self.send_params == "Form-encoded":
            req_headers = {"Accept": accept, "Content-Type": "application/x-www-form-urlencoded"}
            req_body = urllib.parse.urlencode(params, doseq=True)
            r = self.session.post(self._url(path), data=req_body, timeout=self.timeout, headers=req_headers)
        else:  # Query string
            req_headers = {"Accept": accept, "Content-Type": "application/json"}
            req_body = ""
            r = self.session.post(self._url(path, q=params), timeout=self.timeout, headers=req_headers)

        log = AttemptLog(
            auth_mode=self.auth_mode,
            send_params=self.send_params,
            method="POST",
            url=r.request.url,
            status=r.status_code,
            ok=r.ok,
            error="" if r.ok else r.text,
            req_headers=dict(r.request.headers),
            req_body=req_body,
            resp_headers=dict(r.headers),
            resp_body_excerpt=r.text[:800] if r.text else "",
        )
        return r, log

    # ---- Public API ----
    def aggregation_csv(self, params: Dict[str, Any]) -> bytes:
        r, log = self._post("/rest/aggregation", params, "text/csv")
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def aggregation_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        r, log = self._post("/rest/aggregation", params, "application/json")
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def aggregation_json_df(self, params: Dict[str, Any]) -> pd.DataFrame:
        data = self.aggregation_json(params)
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            cols = [c.get("name") if isinstance(c, dict) else str(c) for c in data["columns"]]
            return pd.DataFrame(data["rows"], columns=cols)
        return pd.json_normalize(data)

    # ---- Diagnostics ----
    def matrix_test(self, base_params: Dict[str, Any], override_page_size: int = 1) -> List[AttemptLog]:
        attempts: List[AttemptLog] = []
        sources = []
        if "filterId" in base_params: sources.append(("jqlfilter", {"filterId": base_params["filterId"]}))
        if "projectKey" in base_params: sources.append(("project", {"projectKey": base_params["projectKey"]}))
        if "jql" in base_params: sources.append(("customjql", {"jql": base_params["jql"]}))
        if not sources:
            sources = [("jqlfilter", {"filterId": 1})]

        for auth_mode in ["header", "query"]:
            for send in ["JSON body", "Query string", "Form-encoded"]:
                self.auth_mode = auth_mode
                self.send_params = send
                # Build tiny payload
                params = {
                    "page": 1, "pageSize": override_page_size,
                    "columnsBy": "statusduration",
                    "aggregationType": base_params.get("aggregationType", "average"),
                    "columns": ["Issue Key", "Time in Status (hours)"]
                }
                ftype, fvals = sources[0]
                params["filterType"] = ftype
                params.update(fvals)
                r, log = self._post("/rest/aggregation", params, "application/json")
                attempts.append(log)
        return attempts

    def curl_preview(self, path: str, params: Dict[str, Any], accept: str = "application/json") -> str:
        url = self._url(path) if self.send_params != "Query string" else self._url(path, q=params)
        cmd = ["curl", "-X", "POST", shlex.quote(url), "-H", f"Accept: {accept}"]
        if self.auth_mode == "header":
            cmd += ["-H", f"Authorization: TISJWT {self.token}"]
        if self.send_params == "JSON body":
            cmd += ["-H", "Content-Type: application/json", "--data-binary", shlex.quote(json.dumps(params))]
        elif self.send_params == "Form-encoded":
            cmd += ["-H", "Content-Type: application/x-www-form-urlencoded", "--data", shlex.quote(urllib.parse.urlencode(params))]
        return " ".join(cmd)

    # ---- Extra probes ----
    def probe_calendar(self) -> tuple[bool, str]:
        url = self._url("/calendarSettings/calendar")
        r = self.session.get(url, timeout=self.timeout)
        return (r.ok, r.text)

    # ---- Export (async) helpers ----
    def export_start(self, body: Dict[str, Any]) -> str:
        url = self._url("/report/export")
        r = self.session.post(url, json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        data = r.json()
        return str(data.get("id") or data.get("exportId") or "")

    def export_status(self, export_id: str) -> Dict[str, Any]:
        url = self._url(f"/report/export/{export_id}")
        r = self.session.get(url, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def export_download(self, export_id: str) -> bytes:
        url = self._url(f"/report/export/{export_id}/download")
        r = self.session.get(url, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def export_poll_and_maybe_download(self, interval_sec: int = 2, max_tries: int = 10) -> Dict[str, Any]:
        # naive poller for demo
        last = {}
        for i in range(max_tries):
            time.sleep(interval_sec)
            last = self.export_status(last.get("id", "") if last else "")
        return last
