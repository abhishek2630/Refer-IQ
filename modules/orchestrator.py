"""
Refer IQ — Module F: Case Orchestrator
========================================
Receives a case trigger → runs Modules B–E in parallel where possible →
feeds results into Module A (LLM memo) → assembles and returns the complete
CaseResult object consumed by the FastAPI layer and frontend.

Pipeline:
  1. Load case data from cases.json  (or accept structured dict from API)
  2. PARALLEL: affordability (C) + fraud/AML (D) + RAG (E)
  3. SERIAL:   memo generator (A) — needs C/D/E results as context
  4. Assemble: merge all outputs into CaseResult
  5. Apply policy blocks: AML holds / fraud blocks cap or override the AI limit

Policy blocks applied here (not in individual modules):
  - do_not_issue     → recommended_limit = None, verdict = REFER_SENIOR
  - edd_required     → verdict = EDD_HOLD, limit = post-EDD pending
  - mlro_referral    → adds SAR/MLRO note to regulatory_notes
  - aml_structuring  → adds POCA 2002 note, caps limit to post-AML value

Usage:
    from modules.orchestrator import run_case, CaseResult
    result = run_case("REF-2025-00141")
    print(result.model_dump_json(indent=2))

Run tests (no API key needed — uses mock mode):
    python modules/orchestrator.py --mock

Run live (needs ANTHROPIC_API_KEY in .env):
    python modules/orchestrator.py REF-2025-00138
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ── Import all modules ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.affordability import (
    AffordabilityInput, AffordabilityResult, CommitmentType,
    ExistingCommitment, assess_affordability,
)
from modules.fraud_aml import (
    FraudAMLInput, FraudAMLResult,
    assess_fraud_aml, load_fraud_response,
)
from modules.rag_pipeline import (
    RAGQuery, RAGResult, build_index, query_rag,
)
from modules.memo_generator import (
    CaseInput, CreditMemo, AIVerdict,
    generate_memo,
)


# ── Cases loader ──────────────────────────────────────────────────────────────

def _load_cases(cases_path: str = "synthetic_data/cases.json") -> dict:
    """Load cases.json and return a dict keyed by case_id."""
    path = Path(cases_path)
    if not path.exists():
        raise FileNotFoundError(f"cases.json not found: {cases_path}")
    raw = json.loads(path.read_text())
    cases = raw if isinstance(raw, list) else raw.get("cases", [])
    return {c["id"]: c for c in cases}


# ── Build affordability input from case dict ──────────────────────────────────

def _build_affordability_input(case: dict) -> AffordabilityInput:
    """
    Convert a raw case dict (from cases.json) into an AffordabilityInput.
    Uses pre-computed NDI from open_banking section where available.
    """
    app    = case.get("applicant", {})
    ob     = case.get("open_banking", {})
    bureau = case.get("bureau", {})

    net_monthly  = float(app.get("net_monthly_gbp", 0))
    gross_annual = float(app.get("gross_annual_gbp", 0))

    # Use pre-computed NDI from open banking data to back-calculate total outgoings.
    # NDI = net_monthly - rent - essential_spend - commitments
    # We collapse rent + essential_spend + commitments into essential_spend_monthly
    # so Module C produces the correct NDI without needing itemised breakdown.
    ndi_monthly = float(ob.get("ndi_monthly_gbp") or 0)
    if ndi_monthly > 0 and net_monthly > 0:
        implied_outgoings = net_monthly - ndi_monthly
        essential_spend   = max(implied_outgoings, 0)
    else:
        essential_spend = None   # Module C uses 20% fallback

    credit_util = (
        float(bureau["credit_utilisation_pct"])
        if bureau.get("credit_utilisation_pct") is not None
        else float(ob["credit_utilisation_pct"])
        if ob.get("credit_utilisation_pct") is not None
        else None
    )

    emp_status = app.get("employment_status", "").lower()
    is_self_employed = "self" in emp_status or "director" in emp_status

    return AffordabilityInput(
        gross_monthly=gross_annual / 12,
        net_monthly=net_monthly,
        existing_commitments=[],           # commitments already baked into NDI
        estimated_rent=None,               # baked into essential_spend above
        essential_spend_monthly=essential_spend,
        credit_utilisation_pct=credit_util,
        is_self_employed=is_self_employed,
        is_thin_file=bureau.get("thin_file", False),
        apply_near_prime_cap=True,
    )


# ── Build fraud/AML input from case dict ──────────────────────────────────────

def _build_fraud_input(case: dict) -> FraudAMLInput:
    """
    Try to load from fraud_responses.json first (synthetic API response).
    Fall back to building from the case dict's fraud_aml section.
    """
    case_id = case["id"]
    try:
        inp = load_fraud_response(case_id)
        # Inject open banking AML flags from case data
        ob = case.get("open_banking", {})
        inp.ob_structuring_detected = ob.get("structuring_detected", False)
        inp.ob_gambling_detected    = ob.get("gambling_detected", False)
        inp.ob_payday_detected      = ob.get("payday_loan_detected", False)
        return inp
    except (FileNotFoundError, KeyError):
        pass

    # Fallback: build from case dict
    fa = case.get("fraud_aml", {})
    ob = case.get("open_banking", {})
    return FraudAMLInput(
        case_id=case_id,
        identity_verified=fa.get("identity_verified", True),
        address_match=fa.get("address_match", True),
        address_consistency=fa.get("address_consistency", "confirmed"),
        address_changes_24mo=fa.get("address_changes_24mo", 0),
        pep_match=fa.get("pep_match", False),
        sanctions_match=fa.get("sanctions_match", False),
        declined_12mo=fa.get("declined_12mo", 0),
        hard_searches_6mo=fa.get("hard_searches_6mo", 0),
        device_risk=fa.get("device_risk", "low"),
        ob_structuring_detected=ob.get("structuring_detected", False),
        ob_gambling_detected=ob.get("gambling_detected", False),
        ob_payday_detected=ob.get("payday_loan_detected", False),
    )


# ── Build RAG query from case dict ────────────────────────────────────────────

def _build_rag_query(case: dict, fraud_result: FraudAMLResult) -> RAGQuery:
    """Construct a targeted RAG query from the case signals."""
    case_type = case.get("case_type", "Bureau")
    ob  = case.get("open_banking", {})
    bureau = case.get("bureau", {})

    terms: list[str] = []

    if case_type == "AML":
        if ob.get("structuring_detected"):
            terms += ["cash structuring sub-threshold deposits POCA 2002 SAR suspicious activity"]
        if fraud_result.pep_match if hasattr(fraud_result, "pep_match") else False:
            terms += ["PEP politically exposed person enhanced due diligence EDD MLR 2017"]
        else:
            terms += ["AML structuring money laundering open banking"]

    elif case_type == "Fraud":
        terms += [
            "CIFAS National Hunter fraud address mismatch sequential application",
            "senior underwriter MLRO escalation do not issue",
        ]

    else:  # Bureau
        if bureau.get("thin_file"):
            terms += ["thin file limited credit history Consumer Duty alternative data NHS employment"]
        else:
            terms += [
                "grey band score utilisation Consumer Duty credit limit",
                "FCA CONC stress test NDI near-prime low-to-grow",
            ]

    # Check for PEP flag in fraud result
    for flag in fraud_result.flags:
        if "PEP" in flag.code:
            terms += ["PEP politically exposed person enhanced due diligence EDD MLR 2017"]
            break

    query_text = " ".join(terms)

    return RAGQuery(
        case_id=case["id"],
        case_type=case_type,
        query_text=query_text,
        top_k_policy=4,
        top_k_cases=3,
        filter_case_type=case_type if case_type in ("AML", "Bureau", "Fraud") else None,
    )


# ── Output schema ─────────────────────────────────────────────────────────────

class ProcessingStep(BaseModel):
    """Timing and status for a single pipeline step."""
    name:        str
    status:      str         # "ok" | "error" | "skipped"
    duration_ms: int
    detail:      Optional[str] = None


class CaseResult(BaseModel):
    """
    Complete assembled case output — consumed by FastAPI and frontend.
    This is the single object the frontend replaces its static CASES array with.
    """
    # Identity
    case_id:        str
    applicant_name: str
    processed_at:   str      # ISO-8601 UTC

    # Module outputs
    affordability:  AffordabilityResult
    fraud_aml:      FraudAMLResult
    rag:            RAGResult
    memo:           Optional[CreditMemo] = None    # None if API key not set

    # Final synthesised recommendation (policy blocks applied)
    final_verdict:          str
    final_limit_gbp:        Optional[int]
    final_limit_label:      str          # display string e.g. "£2,000" or "Post-AML clearance"
    policy_blocks_applied:  list[str]    # e.g. ["aml_hold", "near_prime_cap"]

    # Pipeline telemetry
    steps:          list[ProcessingStep]
    total_ms:       int
    pipeline_ok:    bool


# ── Policy block logic ────────────────────────────────────────────────────────

def _apply_policy_blocks(
    case: dict,
    affordability: AffordabilityResult,
    fraud: FraudAMLResult,
    memo: Optional[CreditMemo],
) -> tuple[str, Optional[int], str, list[str]]:
    """
    Apply regulatory and policy blocks on top of module outputs.
    Returns (final_verdict, final_limit_gbp, final_limit_label, blocks_applied).

    Order of precedence (highest to lowest):
      1. Sanctions match         → hard stop
      2. do_not_issue (fraud)    → do not issue, refer senior
      3. edd_required (PEP)      → EDD hold
      4. mlro_referral (AML)     → AML hold, conditional
      5. Affordability red band  → decline
      6. Memo verdict            → follow memo
      7. Affordability result    → follow affordability
    """
    blocks: list[str] = []

    # Block 1: Sanctions
    if any(f.code == "SANCTIONS_MATCH" for f in fraud.flags):
        blocks.append("sanctions_hard_stop")
        return "refer_senior", None, "Hard stop — compliance referral", blocks

    # Block 2: Do not issue (active CIFAS + Hunter, or fraud critical)
    if fraud.do_not_issue:
        blocks.append("fraud_do_not_issue")
        return "refer_senior", None, "Do not issue — senior UW required", blocks

    # Block 3: EDD hold (PEP)
    if fraud.edd_required:
        blocks.append("edd_hold")
        limit = affordability.recommended_limit
        return "edd_hold", limit, f"£{limit:,} — post-EDD clearance", blocks

    # Block 4: AML / structuring hold
    if fraud.mlro_referral_required and any(f.code == "OB_STRUCTURING" for f in fraud.flags):
        blocks.append("aml_structuring_hold")
        # Conservative limit: cap at £1,500 pending AML clearance
        aml_limit = min(affordability.recommended_limit, 1500)
        return "aml_hold", aml_limit, f"£{aml_limit:,} — post-AML clearance", blocks

    # Block 5: Affordability red band
    if affordability.risk_band.value == "red" or not affordability.conc_compliant:
        blocks.append("affordability_decline")
        return "decline", None, "Declined — affordability", blocks

    # Block 6: Follow memo if available
    if memo is not None:
        verdict_str = memo.verdict.value
        limit = memo.recommended_limit_gbp or affordability.recommended_limit
        label = f"£{limit:,}" if limit else "Not issued"
        if memo.limit_condition:
            label += f" — {memo.limit_condition}"
        return verdict_str, limit, label, blocks

    # Block 7: Follow affordability
    limit = affordability.recommended_limit
    verdict = "approve" if affordability.risk_band.value == "green" else "conditional_approve"
    return verdict, limit, f"£{limit:,}", blocks


# ── Main orchestration function ───────────────────────────────────────────────

def run_case(
    case_id: str,
    cases_path: str = "synthetic_data/cases.json",
    fraud_responses_path: str = "synthetic_data/fraud_responses/fraud_responses.json",
    use_memo_generator: bool = True,
    mock_memo: bool = False,
) -> CaseResult:
    """
    Run the full pipeline for a single case.

    Args:
        case_id:            e.g. "REF-2025-00141"
        cases_path:         path to cases.json
        fraud_responses_path: path to fraud_responses.json
        use_memo_generator: if False, skips Module A (no API call)
        mock_memo:          if True, returns a stub CreditMemo (for testing without API key)

    Returns:
        CaseResult with all module outputs assembled
    """
    t_start  = time.time()
    steps:   list[ProcessingStep] = []
    api_key  = os.getenv("ANTHROPIC_API_KEY")

    # ── Load case ─────────────────────────────────────────────────────────────
    t0 = time.time()
    cases = _load_cases(cases_path)
    if case_id not in cases:
        raise KeyError(f"Case {case_id} not found in {cases_path}")
    case = cases[case_id]
    steps.append(ProcessingStep(
        name="load_case", status="ok",
        duration_ms=int((time.time() - t0) * 1000),
    ))

    # ── Ensure RAG index exists ───────────────────────────────────────────────
    t0 = time.time()
    rag_index_path = Path("data/chroma/tfidf_index.pkl")
    if not rag_index_path.exists():
        print("  Building RAG index (first run)…")
        build_index()
    steps.append(ProcessingStep(
        name="rag_index_check", status="ok",
        duration_ms=int((time.time() - t0) * 1000),
    ))

    # ── Step C: Affordability ─────────────────────────────────────────────────
    t0 = time.time()
    try:
        afford_input  = _build_affordability_input(case)
        affordability = assess_affordability(afford_input)
        # Patch DTI from pre-computed ob.dti_pct (more accurate than
        # commitment-based calc when outgoings are collapsed into essential_spend)
        ob_dti = case.get("open_banking", {}).get("dti_pct")
        if ob_dti is not None:
            affordability.dti = round(float(ob_dti) / 100, 4)
            affordability.dti_with_new_card = round(
                affordability.dti + affordability.new_card_min_repayment / (float(case["applicant"]["gross_annual_gbp"]) / 12), 4
            )
        steps.append(ProcessingStep(
            name="affordability", status="ok",
            duration_ms=int((time.time() - t0) * 1000),
            detail=f"NDI £{affordability.ndi:,.0f} | DTI {affordability.dti:.0%} | limit £{affordability.recommended_limit:,} | {affordability.risk_band.value}",
        ))
    except Exception as e:
        steps.append(ProcessingStep(name="affordability", status="error",
                                    duration_ms=int((time.time() - t0) * 1000), detail=str(e)))
        raise

    # ── Step D: Fraud & AML ───────────────────────────────────────────────────
    t0 = time.time()
    try:
        fraud_input = _build_fraud_input(case)
        fraud       = assess_fraud_aml(fraud_input)
        steps.append(ProcessingStep(
            name="fraud_aml", status="ok",
            duration_ms=int((time.time() - t0) * 1000),
            detail=f"score {fraud.fraud_risk_score}/100 | {fraud.fraud_risk_level.value} | {len(fraud.flags)} flags",
        ))
    except Exception as e:
        steps.append(ProcessingStep(name="fraud_aml", status="error",
                                    duration_ms=int((time.time() - t0) * 1000), detail=str(e)))
        raise

    # ── Step E: RAG retrieval ─────────────────────────────────────────────────
    t0 = time.time()
    try:
        rag_query = _build_rag_query(case, fraud)
        rag       = query_rag(rag_query)
        steps.append(ProcessingStep(
            name="rag", status="ok",
            duration_ms=int((time.time() - t0) * 1000),
            detail=f"{len(rag.relevant_policy)} policy chunks | {len(rag.similar_cases)} similar cases",
        ))
    except Exception as e:
        steps.append(ProcessingStep(name="rag", status="error",
                                    duration_ms=int((time.time() - t0) * 1000), detail=str(e)))
        raise

    # ── Step A: Memo generator ────────────────────────────────────────────────
    memo: Optional[CreditMemo] = None

    if mock_memo:
        # Stub for testing without API key
        from modules.memo_generator import (
            RiskFactor, RiskDirection, ConfidenceLevel, AIVerdict
        )
        memo = CreditMemo(
            case_id=case_id,
            applicant_name=case.get("name", ""),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            verdict=AIVerdict.APPROVE,
            confidence=75,
            confidence_level=ConfidenceLevel.HIGH,
            narrative=(
                f"[MOCK MEMO] {case.get('name','')} — "
                f"Affordability: NDI £{affordability.ndi:,.0f}/mo, "
                f"recommended limit £{affordability.recommended_limit:,}. "
                f"Fraud risk: {fraud.fraud_risk_level.value} ({fraud.fraud_risk_score}/100). "
                f"RAG retrieved {len(rag.relevant_policy)} policy chunks."
            ),
            recommended_limit_gbp=affordability.recommended_limit,
            risk_factors=[
                RiskFactor(name="NDI buffer", weight=70,
                           direction=RiskDirection.REDUCING,
                           explanation=f"NDI £{affordability.ndi:,.0f}/mo supports limit"),
                RiskFactor(name="Fraud risk", weight=fraud.fraud_risk_score,
                           direction=RiskDirection.AML_FRAUD if fraud.fraud_risk_score > 40 else RiskDirection.REDUCING,
                           explanation=f"Fraud score {fraud.fraud_risk_score}/100"),
                RiskFactor(name="Credit utilisation", weight=int(affordability.existing_utilisation_pct or 0),
                           direction=RiskDirection.INCREASING if (affordability.existing_utilisation_pct or 0) > 60 else RiskDirection.REDUCING,
                           explanation=f"Utilisation {affordability.existing_utilisation_pct or 0:.0f}%"),
            ],
            regulatory_notes=fraud.explanation,
            fca_conc_pass=affordability.conc_compliant,
            fca_min_repayment_gbp=affordability.new_card_min_repayment,
            fca_repayment_as_pct_ndi=round(affordability.repayment_pct_of_ndi * 100, 1),
        )
        steps.append(ProcessingStep(
            name="memo_generator", status="ok",
            duration_ms=0, detail="mock mode — no API call",
        ))

    elif use_memo_generator and api_key:
        t0 = time.time()
        try:
            case_input = CaseInput(**case)
            memo       = generate_memo(case_input)
            steps.append(ProcessingStep(
                name="memo_generator", status="ok",
                duration_ms=int((time.time() - t0) * 1000),
                detail=f"verdict={memo.verdict.value} | confidence={memo.confidence}%",
            ))
        except Exception as e:
            steps.append(ProcessingStep(
                name="memo_generator", status="error",
                duration_ms=int((time.time() - t0) * 1000),
                detail=str(e),
            ))
            # Non-fatal — orchestrator still returns C/D/E results
    else:
        steps.append(ProcessingStep(
            name="memo_generator", status="skipped",
            duration_ms=0,
            detail="No API key set — add ANTHROPIC_API_KEY to .env to enable",
        ))

    # ── Apply policy blocks ───────────────────────────────────────────────────
    final_verdict, final_limit, limit_label, blocks = _apply_policy_blocks(
        case, affordability, fraud, memo
    )

    total_ms = int((time.time() - t_start) * 1000)

    return CaseResult(
        case_id        = case_id,
        applicant_name = case.get("name", ""),
        processed_at   = datetime.now(timezone.utc).isoformat(),
        affordability  = affordability,
        fraud_aml      = fraud,
        rag            = rag,
        memo           = memo,
        final_verdict          = final_verdict,
        final_limit_gbp        = final_limit,
        final_limit_label      = limit_label,
        policy_blocks_applied  = blocks,
        steps      = steps,
        total_ms   = total_ms,
        pipeline_ok= all(s.status != "error" for s in steps),
    )


# ── CLI test runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Refer IQ — Module F orchestrator")
    parser.add_argument("case_id", nargs="?", default=None,
                        help="Case ID to process (default: run all 5)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock memo generator (no API key needed)")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    use_mock = args.mock or not api_key

    if use_mock and not args.mock:
        print("  Note: ANTHROPIC_API_KEY not set — running in mock mode.")
        print("  Add your key to .env to enable live memo generation.\n")

    ALL_CASES = [
        "REF-2025-00141",
        "REF-2025-00138",
        "REF-2025-00133",
        "REF-2025-00129",
        "REF-2025-00121",
    ]

    cases_to_run = [args.case_id] if args.case_id else ALL_CASES

    print("=" * 70)
    print("Module F — Case Orchestrator — Pipeline test")
    print("=" * 70)

    all_pass = True

    # Expected policy blocks per case
    EXPECTED_BLOCKS = {
        "REF-2025-00141": ["aml_structuring_hold"],
        "REF-2025-00138": [],
        "REF-2025-00133": ["fraud_do_not_issue"],
        "REF-2025-00129": ["edd_hold"],
        "REF-2025-00121": [],
    }

    for case_id in cases_to_run:
        print(f"\n{'─'*68}")
        print(f"  Processing {case_id}…")
        print(f"{'─'*68}")

        try:
            result = run_case(case_id, mock_memo=use_mock)

            # Pipeline steps
            for step in result.steps:
                icon = "✓" if step.status == "ok" else ("⚠" if step.status == "skipped" else "✗")
                detail = f" — {step.detail}" if step.detail else ""
                print(f"  {icon} {step.name:<22} {step.duration_ms:>5}ms{detail}")

            print(f"\n  Affordability:  NDI £{result.affordability.ndi:,.0f}/mo | "
                  f"limit £{result.affordability.recommended_limit:,} | "
                  f"{result.affordability.risk_band.value.upper()}")
            print(f"  Fraud/AML:      score {result.fraud_aml.fraud_risk_score}/100 | "
                  f"{result.fraud_aml.fraud_risk_level.value.upper()} | "
                  f"{len(result.fraud_aml.flags)} flags")
            print(f"  RAG:            {len(result.rag.relevant_policy)} policy | "
                  f"{len(result.rag.similar_cases)} cases | "
                  f"citations: {result.rag.policy_citations}")
            if result.memo:
                print(f"  Memo:           verdict={result.memo.verdict.value} | "
                      f"confidence={result.memo.confidence}%")

            print(f"\n  ┌─ FINAL DECISION ──────────────────────────────────────")
            print(f"  │ Verdict:  {result.final_verdict}")
            print(f"  │ Limit:    {result.final_limit_label}")
            print(f"  │ Blocks:   {result.policy_blocks_applied or 'none'}")
            print(f"  └───────────────────────────────────────────────────────")
            print(f"  Total: {result.total_ms}ms | pipeline_ok={result.pipeline_ok}")

            # Assertions
            expected = EXPECTED_BLOCKS.get(case_id, [])
            blocks_ok = set(expected).issubset(set(result.policy_blocks_applied))
            ok = result.pipeline_ok and blocks_ok
            all_pass = all_pass and ok

            if not blocks_ok:
                print(f"\n  ✗ Block mismatch — expected {expected}, got {result.policy_blocks_applied}")
            else:
                print(f"\n  {'✓ PASS' if ok else '✗ FAIL'}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_pass = False
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Module F — {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 70)
