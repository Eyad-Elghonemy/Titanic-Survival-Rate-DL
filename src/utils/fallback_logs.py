"""
fallback_logs.py

A static snapshot of api_logs, used only when Supabase is unreachable
(e.g. the free-tier project is paused/sleeping). This keeps /logs and
/logs/summary showing real, representative data instead of an empty
state whenever the database happens to be down -- while making it
obvious to the caller (via "service_status": "degraded") that this
isn't live data.

Loaded once at import time from fallback_logs.json, which sits next
to this file.
"""

import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fallback_logs.json")

with open(_PATH, "r") as f:
    FALLBACK_ROWS = json.load(f)
