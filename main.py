from fastapi import FastAPI, HTTPException, Depends
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from collections import defaultdict
from datetime import datetime, timedelta
from src.utils.requests import PassengerData
from src.utils.response import PredictionResponse
from src.utils.config import APP_NAME, VERSION, API_SECRET_KEY, supabase, LogsUnavailableError
from src.utils.inference import predict_survival
from src.utils.fallback_logs import FALLBACK_ROWS

app = FastAPI(title=APP_NAME, version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-KEY")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key


@app.get('/', tags=['check'])
async def home():
    return {
        'message': "Up & Running"
    }


@app.post("/classify", tags=['NN'], response_model=PredictionResponse)
async def classify(passengers: List[PassengerData], api_key: str = Depends(verify_api_key)):
    try:
        response = predict_survival(passengers=passengers)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")


def _fetch_rows(limit: int, order_desc: bool):
    """
    Returns (rows, service_status). Tries Supabase first; on any
    failure (timeout, connection error -- typically a paused/sleeping
    project) falls back to the static FALLBACK_ROWS snapshot instead
    of raising, so /logs and /logs/summary always return 2xx with
    usable data.
    """
    try:
        result = (
            supabase.table("api_logs")
            .select("*")
            .order("created_at", desc=order_desc)
            .limit(limit)
            .execute()
        )
        return result.data, "ok"
    except Exception:
        rows = sorted(FALLBACK_ROWS, key=lambda r: r["created_at"], reverse=order_desc)
        return rows[:limit], "degraded"


@app.get('/logs', tags=['analytics'])
async def get_logs(limit: int = 100, api_key: str = Depends(verify_api_key)):
    """
    IMPORTANT: this endpoint is also used by the frontend's API-key
    gate as a lightweight "is this key valid?" check (a bare GET with
    no side effects). verify_api_key() above has already run by the
    time we get here, so the key is confirmed valid -- a database
    outage from this point on must NOT turn into a non-2xx response,
    or a visitor with a perfectly correct key gets locked out simply
    because Supabase happens to be asleep. So: always return 200 --
    live rows when Supabase is reachable, the FALLBACK_ROWS snapshot
    otherwise.
    """
    rows, status = _fetch_rows(limit, order_desc=True)
    return {"logs": rows, "service_status": status}


def _pick_bucket(span_hours: float) -> str:
    """Pick a sensible time bucket size based on the actual data span."""
    if span_hours <= 3:
        return "5min"
    elif span_hours <= 24 * 3:
        return "hour"
    elif span_hours <= 24 * 90:
        return "day"
    else:
        return "week"


def _bucket_key(dt: datetime, bucket: str) -> str:
    if bucket == "5min":
        floored_minute = (dt.minute // 5) * 5
        return dt.replace(minute=floored_minute, second=0, microsecond=0).isoformat()
    elif bucket == "hour":
        return dt.replace(minute=0, second=0, microsecond=0).isoformat()
    elif bucket == "day":
        return dt.date().isoformat()
    else:  # week
        start_of_week = dt - timedelta(days=dt.weekday())
        return start_of_week.date().isoformat()


_EMPTY_SUMMARY = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "success_rate": 0,
    "total_passengers": 0,
    "total_survived": 0,
    "survival_rate": 0,
    "avg_response_time_ms": None,
    "bucket_size": None,
    "requests_over_time": [],
    "cumulative_requests": [],
}


def _summarize(rows: list, service_status: str) -> dict:
    """Builds the /logs/summary payload from a list of row dicts."""
    if not rows:
        return {**_EMPTY_SUMMARY, "service_status": service_status}

    total_calls = len(rows)
    successful_calls = sum(1 for r in rows if r.get("success"))
    failed_calls = total_calls - successful_calls
    total_passengers = sum(r.get("record_count") or 0 for r in rows)
    total_survived = sum(r.get("survived_count") or 0 for r in rows)
    response_times = [r["response_time_ms"] for r in rows if r.get("response_time_ms") is not None]

    first_dt = datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00"))
    last_dt = datetime.fromisoformat(rows[-1]["created_at"].replace("Z", "+00:00"))
    span_hours = max((last_dt - first_dt).total_seconds() / 3600, 0.01)
    bucket = _pick_bucket(span_hours)

    buckets = defaultdict(int)
    for r in rows:
        dt = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        key = _bucket_key(dt, bucket)
        buckets[key] += 1

    sorted_keys = sorted(buckets.keys())
    requests_over_time = [{"bucket": k, "count": buckets[k]} for k in sorted_keys]

    running_total = 0
    cumulative_requests = []
    for k in sorted_keys:
        running_total += buckets[k]
        cumulative_requests.append({"bucket": k, "total": running_total})

    return {
        "total_calls": total_calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "success_rate": round(successful_calls / total_calls * 100, 1),
        "total_passengers": total_passengers,
        "total_survived": total_survived,
        "survival_rate": round(total_survived / total_passengers * 100, 1) if total_passengers else 0,
        "avg_response_time_ms": round(sum(response_times) / len(response_times)) if response_times else None,
        "bucket_size": bucket,
        "requests_over_time": requests_over_time,
        "cumulative_requests": cumulative_requests,
        "service_status": service_status,
    }


@app.get('/logs/summary', tags=['analytics'])
async def get_logs_summary(api_key: str = Depends(verify_api_key)):
    """
    Returns everything the Analytics dashboard needs, pre-aggregated.
    Same rule as /logs: never a non-2xx response once the key has
    checked out -- falls back to FALLBACK_ROWS when Supabase is down,
    sorted oldest-first to match the bucketing logic below.
    """
    rows, status = _fetch_rows(limit=5000, order_desc=False)
    return _summarize(rows, status)