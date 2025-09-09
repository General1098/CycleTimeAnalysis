# Cycle Time Tracker with Timepiece (v10, diagnostics)

This build adds a **Troubleshooting Console** that tries all permutations of:
- Auth: Header vs Query (`?token=...`)
- Param sending: JSON body, Query string, Form-encoded
- Plus: cURL preview, raw request/response capture, downloadable debug log JSON, calendar probe, and async export tester.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
