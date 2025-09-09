# Cycle Time Tracker (v4) — Robust Timepiece Handling

This build adds safer defaults and better diagnostics for Timepiece API errors:
- **Safe Mode**: limits `pageSize` to 50, uses a minimal column set, and strips unknown columns (often avoids 500 errors).
- **CSV→JSON Fallback**: on 5xx from CSV aggregation, the client retries with JSON to surface useful error details.
- **Test Connection**: quick ping using a small aggregation to validate token/host.
- **Export Fallback**: if aggregation keeps failing, you can kick off an async export and the app will poll & parse.

## Tips to Avoid 500 Errors
1. **Trim columns**: start with only
   - `Issue Key`
   - `Time in Status (hours)`
2. **Small pages**: `pageSize` ≤ 50 while you validate.
3. **Constrain your JQL**: e.g., limit by project and recent updated dates.
4. **Token & Host**: ensure `TISJWT` is current and host is `https://tis.obss.io`.
5. **Try JSON mode**: the app switches automatically and shows details from the error payload.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Secrets
`.streamlit/secrets.toml`:
```toml
[timepiece]
api_host = "https://tis.obss.io"
api_token = "YOUR_TISJWT_TOKEN"
```
