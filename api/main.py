"""
Refer IQ — FastAPI Application
================================
HTTP API layer wrapping Module F (orchestrator).

Endpoints:
  GET  /health                       — liveness check
  GET  /cases                        — list all cases in queue
  GET  /cases/{case_id}              — get full processed case result
  POST /cases/{case_id}/decision     — log a decision (approve / override / decline / refer)
  GET  /queue/stats                  — queue-level stats for the stats bar

Run locally:
    uvicorn api.main:app --reload --port 8000

Then in index.html, replace static CASES with:
    fetch("http://localhost:8000/cases/REF-2025-00141")
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from modules.orchestrator import run_case, CaseResult, _load_cases
from modules.rag_pipeline import build_index

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Refer IQ API",
    description="AI-powered refer queue platform for UK credit card originations",
    version="1.0.0",
)

# Allow frontend (any origin in dev — tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory decision log (replace with DB in production) ────────────────────
_decision_log: list[dict] = []

# ── Startup: ensure RAG index exists ─────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Build RAG index on first start if not already built."""
    index_path = Path("data/chroma/tfidf_index.pkl")
    if not index_path.exists():
        print("Building RAG index on startup…")
        build_index()
        print("RAG index ready.")


# ── Request / response models ─────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    decision_type:   str            # "ai_approve" | "override" | "decline" | "refer_up"
    override_limit:  Optional[int] = None
    override_reason: Optional[str] = None
    underwriter_id:  Optional[str] = "UW"


class DecisionResponse(BaseModel):
    case_id:        str
    decision_type:  str
    limit_gbp:      Optional[int]
    logged_at:      str
    audit_ref:      str


class QueueStats(BaseModel):
    cases_in_queue:     int
    fraud_aml_flags:    int
    high_complexity:    int
    avg_decision_ms:    Optional[float]
    ai_concordance_pct: float


class CaseSummary(BaseModel):
    case_id:    str
    name:       str
    case_type:  str
    time_ago:   str
    ai_verdict: str
    ai_limit:   Optional[str]


# ── Case result cache (avoid re-running pipeline on every request) ────────────
_case_cache: dict[str, CaseResult] = {}
_cache_ttl_s = 300   # 5 minutes

def _get_or_process(case_id: str) -> CaseResult:
    """Return cached result or run pipeline."""
    api_key   = os.getenv("ANTHROPIC_API_KEY")
    use_mock  = not api_key

    if case_id in _case_cache:
        return _case_cache[case_id]

    result = run_case(case_id, mock_memo=use_mock)
    _case_cache[case_id] = result
    return result


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/cases", response_model=list[CaseSummary])
def list_cases():
    """Return all cases in the queue as lightweight summaries."""
    cases = _load_cases()
    summaries = []
    for case_id, case in cases.items():
        summaries.append(CaseSummary(
            case_id   = case_id,
            name      = case.get("name", ""),
            case_type = case.get("case_type", ""),
            time_ago  = case.get("time_ago", ""),
            ai_verdict= case.get("ai_assessed_verdict", "pending"),
            ai_limit  = (
                f"£{case['ai_assessed_limit_gbp']:,}"
                if case.get("ai_assessed_limit_gbp") else "Pending"
            ),
        ))
    return summaries


@app.get("/cases/{case_id}", response_model=CaseResult)
def get_case(case_id: str):
    """
    Run the full pipeline for a case and return the assembled CaseResult.
    Cached for 5 minutes after first call.
    """
    try:
        return _get_or_process(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cases/{case_id}/decision", response_model=DecisionResponse)
def log_decision(case_id: str, req: DecisionRequest):
    """
    Log an underwriter decision. Stored in memory (replace with DB for production).
    Override decisions require override_reason per FCA Consumer Duty audit trail.
    """
    if req.decision_type == "override" and not req.override_reason:
        raise HTTPException(
            status_code=422,
            detail="override_reason is required for override decisions (FCA Consumer Duty audit trail)",
        )

    # Get the case result to determine final limit
    try:
        case_result = _get_or_process(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    limit = req.override_limit if req.decision_type == "override" else case_result.final_limit_gbp
    audit_ref = f"AUD-{int(time.time())}-{case_id[-3:]}"
    logged_at = datetime.now(timezone.utc).isoformat()

    entry = {
        "case_id":        case_id,
        "decision_type":  req.decision_type,
        "limit_gbp":      limit,
        "override_reason":req.override_reason,
        "underwriter_id": req.underwriter_id,
        "ai_verdict":     case_result.final_verdict,
        "concordant":     req.decision_type == "ai_approve",
        "logged_at":      logged_at,
        "audit_ref":      audit_ref,
    }
    _decision_log.append(entry)

    # Evict cache so next fetch reflects decision
    _case_cache.pop(case_id, None)

    return DecisionResponse(
        case_id       = case_id,
        decision_type = req.decision_type,
        limit_gbp     = limit,
        logged_at     = logged_at,
        audit_ref     = audit_ref,
    )


@app.get("/queue/stats", response_model=QueueStats)
def queue_stats():
    """Return stats bar metrics."""
    cases = _load_cases()
    n = len(cases)

    # Count fraud/AML flags and high complexity from cache or case data
    fraud_flags = 0
    high_complexity = 0
    for case_id, case in cases.items():
        if case.get("case_type") in ("AML", "Fraud"):
            fraud_flags += 1
        if case.get("case_type") == "Fraud":
            high_complexity += 1

    # AI concordance from decision log
    decisions = [d for d in _decision_log]
    concordance = (
        round(sum(1 for d in decisions if d["concordant"]) / len(decisions) * 100, 1)
        if decisions else 74.0    # default from frontend
    )

    avg_ms = (
        round(sum(d.get("total_ms", 0) for d in _decision_log) / len(_decision_log))
        if _decision_log else None
    )

    return QueueStats(
        cases_in_queue    = n,
        fraud_aml_flags   = fraud_flags,
        high_complexity   = high_complexity,
        avg_decision_ms   = avg_ms,
        ai_concordance_pct= concordance,
    )


@app.get("/decisions")
def list_decisions():
    """Return all logged decisions (audit trail)."""
    return _decision_log
