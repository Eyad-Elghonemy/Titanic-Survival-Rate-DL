import os
from dotenv import load_dotenv
import joblib
from tensorflow.keras.models import load_model
from supabase import create_client, ClientOptions

# load .env file
load_dotenv(override=True)

# get the variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SRC_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load preprocessor and model
preprocessor = joblib.load(os.path.join(SRC_FOLDER_PATH, 'artifacts', 'preprocessor.joblib'))
model = load_model(os.path.join(SRC_FOLDER_PATH, 'artifacts', 'best_model.keras'))

# Seconds to wait on Supabase before giving up. Short on purpose: a
# paused/sleeping free-tier project can otherwise hang a request for
# a very long time, taking the whole endpoint down with it.
_SUPABASE_TIMEOUT_SECONDS = 5

# supabase client (used for logging API calls to the api_logs table).
# create_client() itself doesn't make a network call, so this is safe
# to construct eagerly even if the project happens to be asleep right
# now -- only the .execute() calls below can actually time out.
supabase = create_client(
    SUPABASE_URL, SUPABASE_KEY,
    options=ClientOptions(postgrest_client_timeout=_SUPABASE_TIMEOUT_SECONDS),
)


class LogsUnavailableError(Exception):
    """Raised when Supabase can't be reached (e.g. project is paused)."""


def log_api_call(operation_type: str, success: bool, record_count: int = 0,
                  survived_count: int = 0, error_detail: str = None,
                  response_time_ms: int = None) -> bool:
    """
    Insert one row into api_logs on a best-effort basis.

    Returns:
        True if the entry was stored, False if Supabase was
        unreachable. Never raises -- a logging failure must not break
        the /classify endpoint that triggered it.
    """
    try:
        supabase.table("api_logs").insert({
            "operation_type": operation_type,
            "success": success,
            "record_count": record_count,
            "survived_count": survived_count,
            "error_detail": error_detail,
            "response_time_ms": response_time_ms,
        }).execute()
        return True
    except Exception as e:
        print(f"[log_api_call] Failed to write log (Supabase unavailable?): {e}")
        return False