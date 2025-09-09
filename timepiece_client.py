from __future__ import annotations
import json, time, urllib.parse, shlex, random
from typing import Dict, Any

import pandas as pd
import requests

class TimepieceClient:
    def __init__(self, api_host: str, token: str, auth_mode: str = "header", send_params: str = "Query string", max_retries:int=5, base_delay:float=1.0, timeout: int = 60):
        self.host = api_host.rstrip("/")
        self.timeout = timeout
        self.auth_mode = auth_mode  # "header" or "query"
        self.send_params = send_params  # "Query string" | "JSON body" | "Form-encoded"
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CycleTimeTracker/1.1 (+retries)"})
        self.token = token.strip()
        if auth_mode == "header":
            self.session.headers.update({"Authorization": f"TISJWT {self.token}"})

    def _url(self, path: str, q: Dict[str, Any] | None = None) -> str:
        url = f"{self.host}{path}"
        qp = q or {}
        if self.auth_mode == "query":
            qp["token"] = self.token
        if qp:
            return url + ("&" if "?" in url else "?") + urllib.parse.urlencode(qp, doseq=True)
        return url

    def _post_once(self, path: str, params: Dict[str, Any], accept: str) -> requests.Response:
        if self.send_params == "JSON body":
            req_headers = {"Accept": accept, "Content-Type": "application/json"}
            req_body = json.dumps(params)
            return self.session.post(self._url(path), data=req_body, timeout=self.timeout, headers=req_headers)
        elif self.send_params == "Form-encoded":
            req_headers = {"Accept": accept, "Content-Type": "application/x-www-form-urlencoded"}
            req_body = urllib.parse.urlencode(params, doseq=True)
            return self.session.post(self._url(path), data=req_body, timeout=self.timeout, headers=req_headers)
        else:  # Query string
            req_headers = {"Accept": accept, "Content-Type": "application/json"}
            return self.session.post(self._url(path, q=params), timeout=self.timeout, headers=req_headers)

    def _post_with_retry(self, path: str, params: Dict[str, Any], accept: str) -> requests.Response:
        attempt = 0
        while True:
            r = self._post_once(path, params, accept)
            if r.status_code != 429:
                return r
            attempt += 1
            if attempt > self.max_retries:
                return r
            # Exponential backoff with jitter
            delay = self.base_delay * (2 ** (attempt - 1))
            delay = delay * (0.7 + 0.6 * random.random())
            time.sleep(delay)

    def aggregation_csv(self, params: Dict[str, Any]) -> bytes:
        r = self._post_with_retry("/rest/aggregation", params, "text/csv")
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def aggregation_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        r = self._post_with_retry("/rest/aggregation", params, "application/json")
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def aggregation_json_df(self, params: Dict[str, Any]) -> pd.DataFrame:
        data = self.aggregation_json(params)
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            cols = [c.get("name") if isinstance(c, dict) else str(c) for c in data["columns"]]
            return pd.DataFrame(data["rows"], columns=cols)
        return pd.json_normalize(data)

    def curl_preview(self, path: str, params: Dict[str, Any], accept: str = "application/json") -> str:
        if self.send_params == "Query string":
            url = self._url(path, q=params)
            hdrs = ["-H", "Accept: " + accept]
            if self.auth_mode == "header":
                hdrs += ["-H", f"Authorization: TISJWT {self.token}"]
            return " ".join(["curl", "-X", "POST", url, *hdrs])
        elif self.send_params == "JSON body":
            url = self._url(path)
            body = json.dumps(params)
            hdrs = ["-H", "Accept: " + accept, "-H", "Content-Type: application/json"]
            if self.auth_mode == "header":
                hdrs += ["-H", f"Authorization: TISJWT {self.token}"]
            return " ".join(["curl", "-X", "POST", url, *hdrs, "--data-binary", json.dumps(params)])
        else:
            url = self._url(path)
            form = urllib.parse.urlencode(params)
            hdrs = ["-H", "Accept: " + accept, "-H", "Content-Type: application/x-www-form-urlencoded"]
            if self.auth_mode == "header":
                hdrs += ["-H", f"Authorization: TISJWT {self.token}"]
            return " ".join(["curl", "-X", "POST", url, *hdrs, "--data", form])
