from __future__ import annotations

from typing import Dict, Any

import requests


class TimepieceClient:
    """
    Minimal OBSS Timepiece client.
    Auth: Authorization: TISJWT <token>
    Host: typically https://tis.obss.io
    """

    def __init__(self, api_host: str, token: str, timeout: int = 60):
        self.api_host = api_host.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"TISJWT {token.strip()}"
        })

    # ---- Aggregation (sync) ----
    def aggregation_csv(self, params: Dict[str, Any]) -> bytes:
        """
        Request aggregated data as CSV bytes.
        """
        url = f"{self.api_host}/rest/aggregation"
        r = self.session.post(url, json=params, timeout=self.timeout, headers={"Accept": "text/csv"})
        if r.status_code >= 400:
            raise RuntimeError(f"Aggregation CSV failed: {r.status_code} {r.text[:500]}")
        return r.content

    def aggregation_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request aggregated data as JSON.
        """
        url = f"{self.api_host}/rest/aggregation"
        r = self.session.post(url, json=params, timeout=self.timeout, headers={"Accept": "application/json"})
        if r.status_code >= 400:
            raise RuntimeError(f"Aggregation JSON failed: {r.status_code} {r.text[:500]}")
        return r.json()

    # ---- Export (async) ----
    def export_start(self, params: Dict[str, Any]) -> str:
        """
        Start an export job. Returns export ID.
        """
        url = f"{self.api_host}/report/export"
        r = self.session.post(url, json=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Export start failed: {r.status_code} {r.text[:500]}")
        data = r.json()
        return str(data.get("id") or data.get("exportId") or "")

    def export_status(self, export_id: str) -> Dict[str, Any]:
        url = f"{self.api_host}/report/export/{export_id}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Export status failed: {r.status_code} {r.text[:500]}")
        return r.json()

    def export_download(self, export_id: str) -> bytes:
        url = f"{self.api_host}/report/export/{export_id}/download"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Export download failed: {r.status_code} {r.text[:500]}")
        return r.content
