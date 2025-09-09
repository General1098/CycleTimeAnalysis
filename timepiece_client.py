from __future__ import annotations
import json, time
from typing import Dict, Any, List, Optional

import requests

class TimepieceClient:
    def __init__(self, api_host: str, token: str, timeout: int = 60):
        self.host = api_host.rstrip("/")
        self.timeout = timeout
        self.token = token.strip()
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"TISJWT {self.token}", "User-Agent": "CycleTimeTracker/1.4 (+export-discovery)"})

        # candidate export endpoints (start/status/download patterns)
        self.export_candidates = [
            { "start": "/report/export", "status": "/report/export/{id}", "download": "/report/export/{id}/download" },
            { "start": "/rest/report/export", "status": "/rest/report/export/{id}", "download": "/rest/report/export/{id}/download" },
            { "start": "/api/report/export", "status": "/api/report/export/{id}", "download": "/api/report/export/{id}/download" },
            { "start": "/rest/obss-tis/latest/report/export", "status": "/rest/obss-tis/latest/report/export/{id}", "download": "/rest/obss-tis/latest/report/export/{id}/download" },
        ]
        self._chosen: Optional[dict] = None

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def discover_export_endpoint(self, sample_params: Dict[str, Any]) -> Optional[str]:
        # Try POST start with minimal body; treat 400/401/429/200 as "exists", 404 as "not here".
        body = {"params": sample_params, "format": "csv"}
        for cand in self.export_candidates:
            try:
                r = self.session.post(self._url(cand["start"]), json=body, timeout=self.timeout)
                if r.status_code != 404:
                    self._chosen = cand
                    return cand["start"]
            except Exception:
                continue
        return None

    def export_start(self, body: Dict[str, Any]) -> str:
        if not self._chosen:
            # lazy discover
            self.discover_export_endpoint(sample_params=body.get("params", {}))
        if not self._chosen:
            raise RuntimeError("No export endpoint found on this host. Try switching API Host to your Jira base (e.g., https://your-domain.atlassian.net).")
        r = self.session.post(self._url(self._chosen["start"]), json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        data = r.json()
        return str(data.get("id") or data.get("exportId") or "")

    def export_status(self, export_id: str) -> Dict[str, Any]:
        if not self._chosen:
            raise RuntimeError("No export endpoint chosen")
        r = self.session.get(self._url(self._chosen["status"].format(id=export_id)), timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def export_download(self, export_id: str) -> bytes:
        if not self._chosen:
            raise RuntimeError("No export endpoint chosen")
        r = self.session.get(self._url(self._chosen["download"].format(id=export_id)), timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def export_flow_fetch_csv(self, params: Dict[str, Any], st=None) -> bytes:
        body = {"params": params, "format": "csv"}
        if st is not None: st.info("Discovering export endpoint…")
        self.discover_export_endpoint(sample_params=params)
        if st is not None and self._chosen: st.info(f"Using {self._chosen['start']}")
        if st is not None: st.info("Starting export…")
        export_id = self.export_start(body)
        if st is not None: st.info(f"Export started: {export_id}. Polling…")
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
