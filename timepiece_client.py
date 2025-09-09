
from __future__ import annotations

import io
import json
from typing import Dict, Any, Optional

import requests


class TimepieceClient:
    """
    Minimal client for OBSS Timepiece Cloud.
    Uses TISJWT token for Authorization.
    """

    def __init__(self, api_host: str, token: str, timeout: int = 60):
        self.api_host = api_host.rstrip("/")
        self.token = token.strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"TISJWT {self.token}"})

    # ---- Aggregation (sync) ----
    def aggregation_csv(self, params: Dict[str, Any]) -> bytes:
        """
        Calls the aggregation endpoint to return data as CSV bytes.
        Endpoint path may vary by tenant/version; default path below works for most cloud setups.
        """
        url = f"{self.api_host}/rest/aggregation"
        r = self.session.post(url, json=params, timeout=self.timeout, headers={"Accept": "text/csv"})
        if r.status_code >= 400:
            raise RuntimeError(f"Aggregation CSV failed: {r.status_code} {r.text[:300]}")
        return r.content

    def aggregation_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_host}/rest/aggregation"
        r = self.session.post(url, json=params, timeout=self.timeout, headers={"Accept": "application/json"})
        if r.status_code >= 400:
            raise RuntimeError(f"Aggregation JSON failed: {r.status_code} {r.text[:300]}")
        return r.json()

    # ---- Async Export flow (optional) ----
    def export_start(self, params: Dict[str, Any]) -> str:
        """
        Starts an export job. Returns export ID.
        """
        url = f"{self.api_host}/report/export"
        r = self.session.post(url, json=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Export start failed: {r.status_code} {r.text[:300]}")
        data = r.json()
        return str(data.get("id") or data.get("exportId") or "")

    def export_status(self, export_id: str) -> Dict[str, Any]:
        url = f"{self.api_host}/report/export/{export_id}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Export status failed: {r.status_code} {r.text[:300]}")
        return r.json()

    def export_download(self, export_id: str) -> bytes:
        url = f"{self.api_host}/report/export/{export_id}/download"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Export download failed: {r.status_code} {r.text[:300]}")
        return r.content
