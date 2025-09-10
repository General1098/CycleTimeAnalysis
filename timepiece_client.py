from __future__ import annotations
import json, time, urllib.parse, shlex
from typing import Dict, Any, Optional, List, Tuple

import requests

class TimepieceClient:
    def __init__(self, api_host: str, token: str, timeout: int = 60):
        self.host = api_host.rstrip("/")
        self.timeout = timeout
        self.token = (token or "").strip()
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"TISJWT {self.token}"})
        self.session.headers.update({"User-Agent": "CycleTimeTracker/1.6 (+manual-override)"})
        self._chosen: Optional[dict] = None
        self._custom: Optional[dict] = None

        # Expanded candidate list (common across tenants)
        self.export_candidates: List[dict] = [
            {"start":"/report/export", "status":"/report/export/{id}", "download":"/report/export/{id}/download"},
            {"start":"/rest/report/export", "status":"/rest/report/export/{id}", "download":"/rest/report/export/{id}/download"},
            {"start":"/api/report/export", "status":"/api/report/export/{id}", "download":"/api/report/export/{id}/download"},
            {"start":"/rest/obss-tis/latest/report/export", "status":"/rest/obss-tis/latest/report/export/{id}", "download":"/rest/obss-tis/latest/report/export/{id}/download"},
            {"start":"/rest/obss-tis/1.0/report/export", "status":"/rest/obss-tis/1.0/report/export/{id}", "download":"/rest/obss-tis/1.0/report/export/{id}/download"},
            # Some tenants place APIs under plugin servlet paths
            {"start":"/plugins/servlet/obss-tis/report/export", "status":"/plugins/servlet/obss-tis/report/export/{id}", "download":"/plugins/servlet/obss-tis/report/export/{id}/download"},
        ]

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def set_custom_paths(self, start: str, status: str, download: str) -> None:
        self._custom = {"start": start, "status": status, "download": download}
        self._chosen = self._custom

    def _start_status_download(self) -> dict:
        if self._chosen: return self._chosen
        return self.export_candidates[0]

    def _exists_status(self, code: int) -> bool:
        # consider anything other than 404 as "endpoint exists"
        return code != 404

    def discover_export_endpoint(self, sample_params: Dict[str, Any]) -> Optional[str]:
        body = {"params": sample_params, "format": "csv"}
        for cand in self.export_candidates:
            try:
                r = self.session.post(self._url(cand["start"]), json=body, timeout=self.timeout)
                if self._exists_status(r.status_code):
                    self._chosen = cand
                    return cand["start"]
            except Exception:
                continue
        return None

    def export_start(self, body: Dict[str, Any]) -> str:
        if not self._chosen:
            self.discover_export_endpoint(sample_params=body.get("params", {}))
        if not self._chosen:
            raise RuntimeError("No export endpoint found on this host. Try switching API Host to your Jira base URL and/or set custom paths.")
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
        if not self._chosen:
            if st: st.info("Discovering export endpoint…")
            self.discover_export_endpoint(sample_params=params)
        if st and self._chosen: st.info(f"Using {self._chosen['start']}")
        if st: st.info("Starting export…")
        export_id = self.export_start(body)
        if st: st.info(f"Export started: {export_id}. Polling…")
        for i in range(30):
            time.sleep(2)
            info = self.export_status(export_id)
            status = str(info.get("status") or info.get("state") or "").lower()
            if status in ("completed","done","finished","ready"):
                if st: st.success("Export ready — downloading…")
                return self.export_download(export_id)
            if status in ("failed","error"):
                raise RuntimeError(f"Export failed: {info}")
        raise RuntimeError("Export timed out")

    # ---- Probes & cURL ----
    def ping_start(self, params: Dict[str, Any]) -> Tuple[int, str]:
        body = {"params": params, "format": "csv"}
        r = self.session.post(self._url(self._start_status_download()["start"]), json=body, timeout=self.timeout)
        return r.status_code, r.text

    def ping_status(self, export_id: str) -> Tuple[int, str]:
        r = self.session.get(self._url(self._start_status_download()["status"].format(id=export_id)), timeout=self.timeout)
        return r.status_code, r.text

    def ping_download(self, export_id: str) -> Tuple[int, str]:
        r = self.session.get(self._url(self._start_status_download()["download"].format(id=export_id)), timeout=self.timeout)
        return r.status_code, r.text

    def curl_preview_start(self, params: Dict[str, Any]) -> str:
        body = json.dumps({"params": params, "format": "csv"})
        url = self._url(self._start_status_download()["start"])
        parts = ["curl -X POST", shlex.quote(url), "-H", "Accept: application/json", "-H", "Content-Type: application/json"]
        if self.token:
            parts += ["-H", f"Authorization: TISJWT {self.token}"]
        parts += ["--data-binary", shlex.quote(body)]
        return " ".join(parts)