import os, time, json, hashlib
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, validator
import httpx

# --- Config ---
METABASE_BASE = os.environ["METABASE_BASE"]           # e.g. https://metabase.company.com
MB_USER = os.environ.get("METABASE_USER")             # service account (if using session auth)
MB_PASS = os.environ.get("METABASE_PASSWORD")
MB_API_KEY = os.environ.get("METABASE_API_KEY")       # optional; if you use API keys
ACTION_BEARER = os.environ["ACTION_BEARER"]           # secret GPT→proxy token
ROW_LIMIT = int(os.environ.get("ROW_LIMIT", "5000"))
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "25"))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "60"))    # seconds

# --- Allowlist (friendly name → card_id & param schema) ---
class DateRange(BaseModel):
    start_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date:   Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")

# class WeeklyRevenueParams(DateRange):
#     product: Optional[str] = Field(None, description="SKU or product code")
#     market:  Optional[str] = Field(None, description="e.g., US, EU, APAC")

# class CohortRetentionParams(DateRange):
#     cohort: Optional[str] = Field(None, description="signup_month like 2025-07")

ALLOWLIST: Dict[str, Dict[str, Any]] = {
    # "weekly_revenue": {
    #     "card_id": 123,                 # your Metabase Saved Question ID
    #     "schema": WeeklyRevenueParams,
    #     "max_days": 370,                # guardrail
    # },
    # "cohort_retention": {
    #     "card_id": 456,
    #     "schema": CohortRetentionParams,
    #     "max_days": 370,
    # },
    "llm_totals": {
        "card_id": 1381,
        "schema": None,  # No parameters needed
        "max_days": None,  # Not applicable - returns all historical data
    },
}

# --- Models ---
class AnswerReq(BaseModel):
    question: str
    params: Optional[Dict[str, Any]] = None

    @validator("question")
    def is_allowlisted(cls, v):
        if v not in ALLOWLIST:
            raise ValueError("Question not allowlisted")
        return v

class AnswerResp(BaseModel):
    summary: str
    columns: List[str]
    rows: List[List[Any]]
    truncated: bool = False
    meta: Dict[str, Any] = {}

# --- App ---
app = FastAPI(
    title="Metabase GPT Proxy",
    description="Secure proxy for ChatGPT to query Metabase data",
    version="1.0.0",
    servers=[
        {"url": "https://mb-gpt-2025-hackathon.onrender.com", "description": "Production"}
    ],
    openapi_url=None  # Disable auto-generated OpenAPI, serve custom schema instead
)

# --- Simple session/cache ---
_session_token: Optional[str] = None
_session_expiry: float = 0.0
_cache: Dict[str, Tuple[float, Any]] = {}

async def get_mb_headers() -> Dict[str, str]:
    # Prefer API key if configured
    if MB_API_KEY:
        return {"x-api-key": MB_API_KEY}
    # Else use a cached session token
    global _session_token, _session_expiry
    if not _session_token or time.time() > _session_expiry:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            r = await client.post(f"{METABASE_BASE}/api/session", json={"username": MB_USER, "password": MB_PASS})
            if r.status_code != 200:
                raise HTTPException(500, f"Metabase auth failed: {r.text}")
            _session_token = r.json()["id"]
            _session_expiry = time.time() + 60 * 50  # refresh ~50m
    return {"X-Metabase-Session": _session_token}

def cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def within_window(p: Dict[str, Any], max_days: int) -> bool:
    # best-effort: if both dates exist, check span
    from datetime import datetime
    s, e = p.get("start_date"), p.get("end_date")
    try:
        if s and e:
            ds = datetime.strptime(s, "%Y-%m-%d")
            de = datetime.strptime(e, "%Y-%m-%d")
            return (de - ds).days <= max_days
    except Exception:
        pass
    return True

def summarize(columns: List[str], rows: List[List[Any]], question: str) -> str:
    if not rows:
        return f"No data returned for {question}."

    # For llm_totals, provide row count and column summary
    if question == "llm_totals":
        return f"Returned {len(rows)} months of data with {len(columns)} metrics including volume, fees, margin, transaction counts across API, Swap, RFQ, Gasless, and Matcha products."

    # For other queries, try to find numeric column for basic stats
    if len(columns) > 1 and len(rows) > 0:
        for col_idx, col in enumerate(columns):
            vals = [r[col_idx] for r in rows if isinstance(r[col_idx], (int, float))]
            if vals:
                return f"{question}: {len(rows)} rows. {col} - min={min(vals):.2f}, max={max(vals):.2f}, avg={sum(vals)/len(vals):.2f}"

    return f"Returned {len(rows)} rows for {question}."

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Metabase GPT Proxy",
        "version": "1.0.0",
        "available_queries": list(ALLOWLIST.keys())
    }

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "ok"}

@app.get("/openapi.json")
async def get_openapi():
    """Serve custom OpenAPI schema"""
    with open("openapi_schema.json", "r") as f:
        return json.load(f)

@app.post("/answer", response_model=AnswerResp, operation_id="queryMetabase")
async def answer(req: AnswerReq, authorization: str = Header(default="")):
    # Auth (GPT → proxy)
    if not authorization.startswith("Bearer ") or authorization.split(" ", 1)[1] != ACTION_BEARER:
        raise HTTPException(401, "Unauthorized")

    cfg = ALLOWLIST[req.question]
    Schema = cfg["schema"]

    # Handle queries without parameters
    if Schema is None:
        params_dict = {}
    else:
        # Handle None params gracefully
        params_input = req.params if req.params is not None else {}
        params_obj = Schema(**params_input)  # pydantic validation
        params_dict = params_obj.dict()

        # extra guardrails
        if cfg.get("max_days") and not within_window(params_dict, cfg["max_days"]):
            raise HTTPException(400, "Date range too large")

    payload = {"question": req.question, "params": params_dict}
    key = cache_key(payload)
    now = time.time()
    if key in _cache and now - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]

    headers = await get_mb_headers()

    # Run Saved Question (Card) with parameters.
    # For parameterized cards, pass parameters by name.
    card_id = cfg["card_id"]
    body = {
        "parameters": [
            {"type": "date", "name": k, "value": v}
            if "date" in k else {"type": "category", "name": k, "value": v}
            for k, v in params_dict.items() if v is not None
        ]
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        r = await client.post(f"{METABASE_BASE}/api/card/{card_id}/query", headers=headers, json=body)
        if r.status_code >= 300:
            raise HTTPException(500, f"Metabase error: {r.text}")
        data = r.json()  # { data: { rows: [...], cols: [...] }, rows_truncated?: bool }

    cols = [c.get("name") for c in data["data"]["cols"]]
    rows = data["data"]["rows"][:ROW_LIMIT]
    truncated = bool(data["data"].get("rows_truncated")) or len(data["data"]["rows"]) > ROW_LIMIT

    resp = AnswerResp(
        summary=summarize(cols, rows, req.question),
        columns=cols,
        rows=rows,
        truncated=truncated,
        meta={"card_id": card_id, "cached": False}
    )
    _cache[key] = (now, resp.dict())
    return resp
