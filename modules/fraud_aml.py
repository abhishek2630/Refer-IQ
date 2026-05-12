"""
Refer IQ — Module D: Fraud & AML Signal Aggregator
====================================================
Takes raw fraud and AML signals (CIFAS, National Hunter, PEP/sanctions,
identity verification, address consistency) and produces a scored,
structured fraud risk assessment.

No API calls — pure signal scoring logic.

Regulation applied:
  - POCA 2002 s.330  — duty to disclose: SAR to NCA if structuring suspected
  - MLR 2017         — EDD mandatory for all PEP relationships
  - CIFAS rules      — Category 6 (misuse) is a hard escalation trigger
  - National Hunter  — 2+ hits in 6 months triggers senior UW review

Input:  FraudAMLInput Pydantic model (or raw dict from fraud_responses.json)
Output: FraudAMLResult Pydantic model

Usage:
    from modules.fraud_aml import assess_fraud_aml, load_fraud_response
    raw = load_fraud_response("REF-2025-00133")
    result = assess_fraud_aml(raw)
    print(result.model_dump_json(indent=2))

Run tests:
    python modules/fraud_aml.py
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ── Scoring weights ───────────────────────────────────────────────────────────
# Each signal contributes points to the fraud risk score (0–100).
# Scores are additive and capped at 100.

SCORE_WEIGHTS = {
    # Hard blockers — high weight, always trigger escalation flags
    "cifas_cat6_active":          85,   # active CIFAS Category 6
    "cifas_cat6_discharged":      55,   # discharged CIFAS Cat 6 — still significant
    "cifas_cat1_5":               40,   # other CIFAS categories (1–5)
    "sanctions_match":            95,   # sanctions list match — hard stop
    "pep_match":                  30,   # PEP match — requires EDD, not a decline

    # Identity and address
    "address_mismatch":           25,   # app vs bureau postcode mismatch
    "address_changes_3plus_24mo": 20,   # 3+ address changes in 24 months
    "identity_address_fail":      20,   # identity check address failed

    # Application behaviour
    "hunter_hits_1":              20,   # 1 National Hunter hit
    "hunter_hits_2plus":          40,   # 2+ National Hunter hits
    "declines_1":                 10,   # 1 prior decline in 12 months
    "declines_2plus":             25,   # 2+ prior declines in 12 months
    "hard_searches_5plus":        10,   # 5+ hard searches in 6 months

    # Device / channel
    "vpn_detected":               10,
    "proxy_detected":             15,
    "device_risk_high":           20,
    "device_risk_medium":          8,

    # Open banking AML signals (passed in from Module B doc_parser flags)
    "structuring_detected":       70,   # POCA 2002 s.330 obligation -- critical by default   # sub-threshold cash deposits
    "gambling_detected":           8,
    "payday_loan_detected":       15,
}

# Score thresholds
SCORE_LOW    = 20   # below this = low risk
SCORE_MEDIUM = 45   # below this = medium risk
SCORE_HIGH   = 65   # below this = high risk; at or above = critical


# ── Input schema ──────────────────────────────────────────────────────────────

class CIFASMarker(BaseModel):
    category: int
    category_description: str
    filing_date: str
    status: str                          # "active" | "discharged"
    discharge_date: Optional[str] = None
    case_ref: Optional[str] = None


class HunterHit(BaseModel):
    hit_date: str
    lender_type: Optional[str] = None
    outcome: Optional[str] = None
    postcode_used: Optional[str] = None
    name_used: Optional[str] = None


class PEPMatchDetail(BaseModel):
    matched_name: str
    matched_dob: Optional[str] = None
    match_type: str                      # "name_only" | "name_and_dob" | "full_match"
    pep_category: str                    # "domestic_pep" | "foreign_pep" | "international_org"
    pep_role: Optional[str] = None
    pep_jurisdiction: Optional[str] = None
    match_confidence: str                # "low" | "medium" | "high"
    match_notes: Optional[str] = None


class FraudAMLInput(BaseModel):
    """Structured fraud and AML signals for a single case."""
    case_id: str

    # CIFAS
    cifas_markers:          list[CIFASMarker] = Field(default_factory=list)

    # National Hunter
    hunter_hits:            list[HunterHit]   = Field(default_factory=list)

    # Identity
    identity_verified:      bool = True
    address_match:          bool = True
    address_match_detail:   Optional[str] = None

    # Address consistency
    application_postcode:   Optional[str] = None
    bureau_postcode:        Optional[str] = None
    address_changes_24mo:   int = 0
    address_consistency:    str = "confirmed"   # "confirmed" | "mismatch" | "unverified"

    # PEP / sanctions
    pep_match:              bool = False
    pep_match_detail:       Optional[PEPMatchDetail] = None
    sanctions_match:        bool = False

    # Prior applications
    declined_12mo:          int = 0
    hard_searches_6mo:      int = 0

    # Device intelligence
    vpn_detected:           bool = False
    proxy_detected:         bool = False
    device_risk:            str = "low"          # "low" | "medium" | "high"

    # Open banking AML flags (from Module B)
    ob_structuring_detected: bool = False
    ob_gambling_detected:    bool = False
    ob_payday_detected:      bool = False


# ── Output schema ─────────────────────────────────────────────────────────────

class FraudRiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class FraudFlag(BaseModel):
    """A single identified fraud or AML concern."""
    code:        str           # machine-readable flag code
    label:       str           # human-readable label
    severity:    str           # "info" | "warning" | "high" | "critical"
    detail:      Optional[str] = None
    regulation:  Optional[str] = None   # e.g. "POCA 2002 s.330", "MLR 2017"


class FraudAMLResult(BaseModel):
    """
    Output of the fraud and AML signal aggregator.
    Consumed by Module A (memo generator) and Module F (orchestrator).
    """
    case_id:                str
    fraud_risk_score:       int    = Field(ge=0, le=100)
    fraud_risk_level:       FraudRiskLevel
    flags:                  list[FraudFlag] = Field(default_factory=list)

    # Key decision outputs consumed by Module F
    mlro_referral_required: bool   # POCA 2002 / MLR 2017 obligation
    edd_required:           bool   # MLR 2017 — PEP relationship
    senior_uw_required:     bool   # case exceeds junior analyst authority
    do_not_issue:           bool   # card must not be issued under any circumstances
    sar_consideration:      bool   # SAR to NCA under POCA 2002 should be considered

    # Summary for memo generator
    explanation:            str
    recommended_action:     str


# ── Signal scoring ────────────────────────────────────────────────────────────

def _score_and_flag(inp: FraudAMLInput) -> tuple[int, list[FraudFlag]]:
    """
    Score the case and collect flags. Returns (raw_score, flags).
    Score is uncapped — caller caps at 100.
    """
    score = 0
    flags: list[FraudFlag] = []

    # ── CIFAS ────────────────────────────────────────────────────────────────
    for marker in inp.cifas_markers:
        if marker.category == 6:
            if marker.status == "active":
                score += SCORE_WEIGHTS["cifas_cat6_active"]
                flags.append(FraudFlag(
                    code="CIFAS_CAT6_ACTIVE",
                    label=f"CIFAS Category 6 — {marker.category_description} (ACTIVE)",
                    severity="critical",
                    detail=f"Filed {marker.filing_date}. Ref: {marker.case_ref}",
                    regulation="CIFAS rules — Category 6 active marker is a hard decline signal",
                ))
            else:
                score += SCORE_WEIGHTS["cifas_cat6_discharged"]
                flags.append(FraudFlag(
                    code="CIFAS_CAT6_DISCHARGED",
                    label=f"CIFAS Category 6 — {marker.category_description} (discharged {marker.discharge_date})",
                    severity="high",
                    detail=f"Filed {marker.filing_date}, discharged {marker.discharge_date}. Ref: {marker.case_ref}",
                    regulation="CIFAS rules — discharged Cat 6 remains a significant risk indicator",
                ))
        elif 1 <= marker.category <= 5:
            score += SCORE_WEIGHTS["cifas_cat1_5"]
            flags.append(FraudFlag(
                code=f"CIFAS_CAT{marker.category}",
                label=f"CIFAS Category {marker.category} — {marker.category_description}",
                severity="high",
                detail=f"Filed {marker.filing_date}. Status: {marker.status}",
                regulation="CIFAS rules",
            ))

    # ── Sanctions ────────────────────────────────────────────────────────────
    if inp.sanctions_match:
        score += SCORE_WEIGHTS["sanctions_match"]
        flags.append(FraudFlag(
            code="SANCTIONS_MATCH",
            label="Sanctions list match",
            severity="critical",
            detail="Applicant name/DOB matched on UK or international sanctions database.",
            regulation="UK Sanctions and Anti-Money Laundering Act 2018 — hard stop, do not process",
        ))

    # ── PEP ──────────────────────────────────────────────────────────────────
    if inp.pep_match:
        score += SCORE_WEIGHTS["pep_match"]
        detail = None
        if inp.pep_match_detail:
            d = inp.pep_match_detail
            detail = (
                f"Match type: {d.match_type}. Role: {d.pep_role}. "
                f"Jurisdiction: {d.pep_jurisdiction}. Confidence: {d.match_confidence}. "
                f"{d.match_notes or ''}"
            )
        flags.append(FraudFlag(
            code="PEP_MATCH",
            label="PEP screening match — Enhanced Due Diligence required",
            severity="high",
            detail=detail,
            regulation="MLR 2017 Reg. 33 — EDD mandatory for all PEP relationships",
        ))

    # ── National Hunter ──────────────────────────────────────────────────────
    hit_count = len(inp.hunter_hits)
    if hit_count == 1:
        score += SCORE_WEIGHTS["hunter_hits_1"]
        flags.append(FraudFlag(
            code="HUNTER_HIT_1",
            label="National Hunter — 1 hit",
            severity="warning",
            detail=f"Hit dated {inp.hunter_hits[0].hit_date}. Lender type: {inp.hunter_hits[0].lender_type}.",
        ))
    elif hit_count >= 2:
        score += SCORE_WEIGHTS["hunter_hits_2plus"]
        dates = ", ".join(h.hit_date for h in inp.hunter_hits)
        postcodes = list({h.postcode_used for h in inp.hunter_hits if h.postcode_used})
        detail = f"{hit_count} hits on {dates}."
        if len(postcodes) > 1:
            detail += f" Multiple postcodes used: {', '.join(postcodes)} — possible sequential application fraud."
        flags.append(FraudFlag(
            code="HUNTER_HITS_MULTIPLE",
            label=f"National Hunter — {hit_count} hits (6 months)",
            severity="high",
            detail=detail,
            regulation="National Hunter guidance — 2+ hits triggers senior UW review",
        ))

    # ── Address consistency ──────────────────────────────────────────────────
    if inp.address_consistency == "mismatch" or not inp.address_match:
        score += SCORE_WEIGHTS["address_mismatch"]
        detail = inp.address_match_detail or (
            f"Application postcode ({inp.application_postcode}) does not match "
            f"bureau postcode ({inp.bureau_postcode})"
        )
        flags.append(FraudFlag(
            code="ADDRESS_MISMATCH",
            label="Address mismatch — application vs bureau",
            severity="high",
            detail=detail,
        ))

    if inp.address_changes_24mo >= 3:
        score += SCORE_WEIGHTS["address_changes_3plus_24mo"]
        flags.append(FraudFlag(
            code="ADDRESS_INSTABILITY",
            label=f"Address instability — {inp.address_changes_24mo} changes in 24 months",
            severity="warning",
            detail="Frequent address changes combined with other flags can indicate synthetic identity creation.",
        ))

    # ── Prior declines ───────────────────────────────────────────────────────
    if inp.declined_12mo == 1:
        score += SCORE_WEIGHTS["declines_1"]
        flags.append(FraudFlag(
            code="PRIOR_DECLINE_1",
            label="1 prior decline in 12 months",
            severity="info",
        ))
    elif inp.declined_12mo >= 2:
        score += SCORE_WEIGHTS["declines_2plus"]
        flags.append(FraudFlag(
            code="PRIOR_DECLINES_MULTIPLE",
            label=f"{inp.declined_12mo} prior declines in 12 months",
            severity="high",
            detail="Multiple recent declines combined with Hunter hits is consistent with sequential application fraud.",
        ))

    # Hard searches
    if inp.hard_searches_6mo >= 5:
        score += SCORE_WEIGHTS["hard_searches_5plus"]
        flags.append(FraudFlag(
            code="HARD_SEARCHES_HIGH",
            label=f"{inp.hard_searches_6mo} hard searches in 6 months",
            severity="warning",
        ))

    # ── Device intelligence ──────────────────────────────────────────────────
    if inp.vpn_detected:
        score += SCORE_WEIGHTS["vpn_detected"]
        flags.append(FraudFlag(
            code="VPN_DETECTED",
            label="VPN detected at application",
            severity="warning",
        ))
    if inp.proxy_detected:
        score += SCORE_WEIGHTS["proxy_detected"]
        flags.append(FraudFlag(
            code="PROXY_DETECTED",
            label="Proxy server detected at application",
            severity="high",
        ))
    if inp.device_risk == "high":
        score += SCORE_WEIGHTS["device_risk_high"]
        flags.append(FraudFlag(
            code="DEVICE_RISK_HIGH",
            label="High-risk device fingerprint",
            severity="high",
        ))
    elif inp.device_risk == "medium":
        score += SCORE_WEIGHTS["device_risk_medium"]
        flags.append(FraudFlag(
            code="DEVICE_RISK_MEDIUM",
            label="Medium-risk device fingerprint",
            severity="info",
        ))

    # ── Open banking AML signals ─────────────────────────────────────────────
    if inp.ob_structuring_detected:
        score += SCORE_WEIGHTS["structuring_detected"]
        flags.append(FraudFlag(
            code="OB_STRUCTURING",
            label="Possible cash structuring detected in open banking data",
            severity="critical",
            detail="Sub-threshold cash deposits detected across consecutive months. Possible smurfing pattern under POCA 2002.",
            regulation="POCA 2002 s.330 — duty to disclose if structuring suspected. SAR to NCA may be required.",
        ))

    if inp.ob_gambling_detected:
        score += SCORE_WEIGHTS["gambling_detected"]
        flags.append(FraudFlag(
            code="OB_GAMBLING",
            label="Gambling transactions detected in open banking data",
            severity="info",
            detail="Gambling activity noted — not a fraud signal but relevant to affordability assessment.",
        ))

    if inp.ob_payday_detected:
        score += SCORE_WEIGHTS["payday_loan_detected"]
        flags.append(FraudFlag(
            code="OB_PAYDAY_LOAN",
            label="Payday loan activity detected in open banking data",
            severity="warning",
            detail="Payday lending activity indicates financial stress not fully captured by bureau data.",
        ))

    return score, flags


def _build_explanation(
    inp: FraudAMLInput,
    score: int,
    risk_level: FraudRiskLevel,
    flags: list[FraudFlag],
) -> tuple[str, str]:
    """Build human-readable explanation and recommended action strings."""

    flag_labels = [f.label for f in flags if f.severity in ("critical", "high")]
    critical_flags = [f for f in flags if f.severity == "critical"]
    high_flags     = [f for f in flags if f.severity == "high"]

    if not flags:
        explanation = (
            f"All fraud and AML checks clear. No CIFAS markers, no National Hunter hits, "
            f"identity verified, address consistent. Fraud risk score: {score}/100 ({risk_level.value})."
        )
        action = "No fraud or AML concerns. Proceed with credit assessment."
        return explanation, action

    parts = [f"Fraud risk score: {score}/100 ({risk_level.value.upper()})."]
    if flag_labels:
        parts.append(f"Key concerns: {'; '.join(flag_labels[:4])}.")

    if inp.sanctions_match:
        parts.append("SANCTIONS MATCH — this case must not proceed. Refer to compliance immediately.")
    elif inp.cifas_markers and any(m.category == 6 for m in inp.cifas_markers):
        if any(m.status == "active" for m in inp.cifas_markers if m.category == 6):
            parts.append("Active CIFAS Category 6 marker — card cannot be issued. Escalate to senior underwriter and MLRO.")
        else:
            parts.append("Discharged CIFAS Category 6 marker — significant fraud history requires senior underwriter sign-off.")
    elif inp.pep_match and not inp.sanctions_match and not inp.cifas_markers:
        parts.append("PEP match requires Enhanced Due Diligence (MLR 2017) before any credit decision. This is a compliance hold, not a decline.")
    elif inp.ob_structuring_detected:
        parts.append("Cash structuring pattern requires AML analyst review and possible SAR submission (POCA 2002 s.330).")

    explanation = " ".join(parts)

    # Recommended action
    if inp.sanctions_match:
        action = "HARD STOP — do not issue card. Refer to Compliance immediately. Do not inform applicant of reason."
    elif inp.cifas_markers and any(m.category == 6 and m.status == "active" for m in inp.cifas_markers):
        action = "Do not issue card. Escalate to senior underwriter and MLRO with full fraud case file."
    elif len(high_flags) >= 2 or (inp.cifas_markers and inp.hunter_hits):
        action = "Refer to senior underwriter. Multiple fraud indicators — do not decide at junior analyst level."
    elif inp.pep_match:
        action = "Place case on hold. Initiate Enhanced Due Diligence (EDD) process. Escalate to Compliance / MLRO. Expected completion: 5–14 working days."
    elif inp.ob_structuring_detected:
        action = "Refer to AML analyst. Request source-of-funds documentation. Consider SAR submission to NCA under POCA 2002 s.330."
    elif risk_level == FraudRiskLevel.MEDIUM:
        action = "Proceed with enhanced scrutiny. Document rationale for approval. Flag for post-decision review."
    else:
        action = "Low fraud risk — proceed with standard credit assessment."

    return explanation, action


# ── Main entry point ──────────────────────────────────────────────────────────

def assess_fraud_aml(inp: FraudAMLInput) -> FraudAMLResult:
    """
    Run the full fraud and AML assessment and return a FraudAMLResult.
    This is the primary entry point for Module D.
    """
    raw_score, flags = _score_and_flag(inp)
    score = min(raw_score, 100)

    if score < SCORE_LOW:
        risk_level = FraudRiskLevel.LOW
    elif score < SCORE_MEDIUM:
        risk_level = FraudRiskLevel.MEDIUM
    elif score < SCORE_HIGH:
        risk_level = FraudRiskLevel.HIGH
    else:
        risk_level = FraudRiskLevel.CRITICAL

    # Decision outputs
    mlro_referral_required = (
        inp.sanctions_match
        or inp.pep_match
        or inp.ob_structuring_detected
        or any(m.category == 6 and m.status == "active" for m in inp.cifas_markers)
        # Discharged Cat6 + multiple Hunter hits = pattern of repeat offending -> MLRO
        or (any(m.category == 6 for m in inp.cifas_markers) and len(inp.hunter_hits) >= 2)
    )

    edd_required = inp.pep_match or inp.sanctions_match

    senior_uw_required = (
        mlro_referral_required
        or len(inp.cifas_markers) > 0
        or len(inp.hunter_hits) >= 2
        or score >= SCORE_HIGH
    )

    do_not_issue = (
        inp.sanctions_match
        or any(m.category == 6 and m.status == "active" for m in inp.cifas_markers)
        or (len(inp.cifas_markers) > 0 and len(inp.hunter_hits) >= 2)
    )

    sar_consideration = inp.ob_structuring_detected or inp.sanctions_match

    explanation, recommended_action = _build_explanation(inp, score, risk_level, flags)

    return FraudAMLResult(
        case_id=inp.case_id,
        fraud_risk_score=score,
        fraud_risk_level=risk_level,
        flags=flags,
        mlro_referral_required=mlro_referral_required,
        edd_required=edd_required,
        senior_uw_required=senior_uw_required,
        do_not_issue=do_not_issue,
        sar_consideration=sar_consideration,
        explanation=explanation,
        recommended_action=recommended_action,
    )


# ── Helper: load from synthetic data ─────────────────────────────────────────

def load_fraud_response(
    case_id: str,
    fraud_responses_path: str = "synthetic_data/fraud_responses/fraud_responses.json",
) -> FraudAMLInput:
    """
    Load a raw fraud response dict from the synthetic data file and
    parse it into a FraudAMLInput model.
    """
    path = Path(fraud_responses_path)
    if not path.exists():
        raise FileNotFoundError(f"Fraud responses file not found: {fraud_responses_path}")

    with open(path) as f:
        all_responses = json.load(f)

    if case_id not in all_responses:
        raise KeyError(f"Case {case_id} not found in fraud responses")

    raw = all_responses[case_id]
    cifas_raw   = raw.get("cifas", {})
    hunter_raw  = raw.get("national_hunter", {})
    identity    = raw.get("identity", {})
    address     = raw.get("address", {})
    pep_raw     = raw.get("pep_sanctions", {})
    prior       = raw.get("prior_applications", {})
    device      = raw.get("device_intelligence", {})

    cifas_markers = [CIFASMarker(**m) for m in cifas_raw.get("markers", [])]
    hunter_hits   = [HunterHit(**h)   for h in hunter_raw.get("hits", [])]

    pep_detail = None
    if pep_raw.get("pep_match") and pep_raw.get("pep_match_detail"):
        pep_detail = PEPMatchDetail(**pep_raw["pep_match_detail"])

    return FraudAMLInput(
        case_id=case_id,
        cifas_markers=cifas_markers,
        hunter_hits=hunter_hits,
        identity_verified=identity.get("kb_pass", True) and identity.get("doc_check_pass", True),
        address_match=identity.get("address_match", True),
        address_match_detail=identity.get("address_match_detail"),
        application_postcode=address.get("application_postcode"),
        bureau_postcode=address.get("bureau_postcode"),
        address_changes_24mo=address.get("address_changes_24mo", 0),
        address_consistency=address.get("consistency", "confirmed"),
        pep_match=pep_raw.get("pep_match", False),
        pep_match_detail=pep_detail,
        sanctions_match=pep_raw.get("sanctions_match", False),
        declined_12mo=prior.get("declined_12mo", 0),
        hard_searches_6mo=prior.get("total_searches_6mo", 0),
        vpn_detected=device.get("vpn_detected", False),
        proxy_detected=device.get("proxy_detected", False),
        device_risk=device.get("device_risk", "low"),
    )


# ── CLI test runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Module D — Fraud & AML Signal Aggregator — All 5 cases")
    print("=" * 70)

    EXPECTED = {
        "REF-2025-00141": {
            "risk_level": FraudRiskLevel.CRITICAL,   # structuring
            "mlro": True, "edd": False, "senior": True, "dni": False, "sar": True,
        },
        "REF-2025-00138": {
            "risk_level": FraudRiskLevel.LOW,
            "mlro": False, "edd": False, "senior": False, "dni": False, "sar": False,
        },
        "REF-2025-00133": {
            "risk_level": FraudRiskLevel.CRITICAL,   # CIFAS Cat6 + 2 Hunter hits + address mismatch
            "mlro": True, "edd": False, "senior": True, "dni": True, "sar": False,
        },
        "REF-2025-00129": {
            "risk_level": FraudRiskLevel.MEDIUM,     # PEP match only
            "mlro": True, "edd": True, "senior": True, "dni": False, "sar": False,
        },
        "REF-2025-00121": {
            "risk_level": FraudRiskLevel.LOW,
            "mlro": False, "edd": False, "senior": False, "dni": False, "sar": False,
        },
    }

    # Inject open banking structuring flags for Mellor (would come from Module B)
    OB_FLAGS = {
        "REF-2025-00141": {"ob_structuring_detected": True,  "ob_gambling_detected": False},
        "REF-2025-00133": {"ob_structuring_detected": False, "ob_gambling_detected": True},
    }

    all_pass = True

    for case_id, exp in EXPECTED.items():
        print(f"\n{'─'*68}")
        print(f"  {case_id}")
        print(f"{'─'*68}")

        inp = load_fraud_response(case_id)

        # Inject OB flags if present
        if case_id in OB_FLAGS:
            for k, v in OB_FLAGS[case_id].items():
                setattr(inp, k, v)

        result = assess_fraud_aml(inp)

        print(f"  Fraud risk score:   {result.fraud_risk_score}/100")
        print(f"  Risk level:         {result.fraud_risk_level.value.upper()}")
        print(f"  MLRO referral:      {'Yes' if result.mlro_referral_required else 'No'}")
        print(f"  EDD required:       {'Yes' if result.edd_required else 'No'}")
        print(f"  Senior UW required: {'Yes' if result.senior_uw_required else 'No'}")
        print(f"  Do not issue:       {'YES ⛔' if result.do_not_issue else 'No'}")
        print(f"  SAR consideration:  {'Yes ⚠' if result.sar_consideration else 'No'}")

        if result.flags:
            print(f"\n  Flags ({len(result.flags)}):")
            for f in result.flags:
                icon = {"critical": "🔴", "high": "🟠", "warning": "🟡", "info": "ℹ"}.get(f.severity, "•")
                print(f"    {icon} [{f.severity.upper()}] {f.label}")
                if f.regulation:
                    print(f"       Reg: {f.regulation}")
        else:
            print(f"\n  ✓ No flags — all checks clear")

        print(f"\n  Action: {result.recommended_action}")

        # Assertions
        ok_level  = result.fraud_risk_level  == exp["risk_level"]
        ok_mlro   = result.mlro_referral_required == exp["mlro"]
        ok_edd    = result.edd_required           == exp["edd"]
        ok_senior = result.senior_uw_required     == exp["senior"]
        ok_dni    = result.do_not_issue           == exp["dni"]
        ok_sar    = result.sar_consideration      == exp["sar"]
        case_ok   = all([ok_level, ok_mlro, ok_edd, ok_senior, ok_dni, ok_sar])
        all_pass  = all_pass and case_ok

        checks = [
            ("Risk level",  ok_level,  f"{result.fraud_risk_level.value} vs {exp['risk_level'].value}"),
            ("MLRO",        ok_mlro,   ""),
            ("EDD",         ok_edd,    ""),
            ("Senior UW",   ok_senior, ""),
            ("Do not issue",ok_dni,    ""),
            ("SAR",         ok_sar,    ""),
        ]
        print(f"\n  {'✓ PASS' if case_ok else '✗ FAIL'}  |  " +
              "  ".join(f"{'✓' if ok else '✗'} {lbl}" + (f" ({detail})" if not ok and detail else "")
                        for lbl, ok, detail in checks))

    print(f"\n{'='*70}")
    print(f"Module D — {'PASS ✓' if all_pass else 'FAIL ✗'} — all assertions checked")
    print("=" * 70)
