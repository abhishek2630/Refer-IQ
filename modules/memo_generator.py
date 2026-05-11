"""
Module A — AI Credit Memo Generator
====================================
Refer IQ · UK credit card originations · near-prime strategy

Takes a structured case dict (from cases.json) → calls Claude API →
returns a validated CreditMemo Pydantic model.

Production pattern:
  - Pydantic v2 models for input validation and output schema enforcement
  - Anthropic SDK (not raw HTTP) with retry logic
  - Structured JSON output prompted via XML tags + JSON schema hint
  - System prompt encodes UK credit policy + FCA CONC rules
  - All monetary values in GBP pence (int) internally; display helpers convert
  - Logging via structlog for production observability
  - No hardcoded API key — loaded from environment via python-dotenv

Usage:
    from modules.memo_generator import generate_memo, CaseInput
    import json

    with open("synthetic_data/cases.json") as f:
        cases = json.load(f)["cases"]

    case = CaseInput(**cases[0])
    memo = generate_memo(case)
    print(memo.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("refer_iq.memo_generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0

# FCA CONC 5.2 — minimum repayment as % of credit limit (floor 2%)
FCA_MIN_REPAYMENT_RATE = 0.02
# Maximum acceptable NDI spend on new minimum repayment (25% is a common policy threshold)
FCA_MAX_NDI_REPAYMENT_PCT = 25.0


# ---------------------------------------------------------------------------
# Input schema  (mirrors cases.json structure)
# ---------------------------------------------------------------------------

class ApplicantInput(BaseModel):
    employment_status: str
    employer: str
    tenure_months: int
    gross_annual_gbp: float
    net_monthly_gbp: float
    gross_annual_gbp_source: Optional[str] = None
    net_monthly_gbp_note: Optional[str] = None


class BureauInput(BaseModel):
    cra: str
    score: int
    score_max: int
    score_band: str
    ccjs_6yr: int = 0
    ccjs_detail: Optional[str] = None
    defaults_6yr: int = 0
    defaults_detail: Optional[str] = None
    arrears_12mo: int = 0
    arrears_detail: Optional[str] = None
    hard_searches_6mo: int
    electoral_roll: Optional[bool] = None
    electoral_roll_years: Optional[int] = None
    credit_utilisation_pct: Optional[float] = None
    active_revolving_accounts: Optional[int] = None
    thin_file: Optional[bool] = False
    thin_file_detail: Optional[str] = None
    limited_uk_history: Optional[bool] = False
    limited_uk_history_detail: Optional[str] = None
    active_revolving_detail: Optional[str] = None


class FraudAmlInput(BaseModel):
    cifas_marker: bool
    cifas_category: Optional[int] = None
    cifas_detail: Optional[str] = None
    national_hunter_hits: int = 0
    national_hunter_detail: Optional[str] = None
    identity_verified: bool
    identity_method: Optional[str] = None
    address_consistent: bool
    address_changes_18mo: Optional[int] = None
    address_stable_years: Optional[int] = None
    app_bureau_address_match: bool
    app_bureau_address_mismatch_detail: Optional[str] = None
    prior_declines_12mo: int = 0
    pep_match: bool
    pep_detail: Optional[str] = None
    sanctions_match: bool
    aml_flag: bool
    aml_flag_detail: Optional[str] = None


class OpenBankingInput(BaseModel):
    data_days: int
    data_days_note: Optional[str] = None
    avg_net_credits_monthly_gbp: float
    salary_consistent: bool
    pay_frequency: str
    structuring_detected: bool
    structuring_detail: Optional[str] = None
    gambling_activity: bool
    gambling_monthly_avg_gbp: float = 0.0
    payday_loan_activity: bool
    ndi_monthly_gbp: float
    ndi_monthly_gbp_note: Optional[str] = None
    dti_pct: float
    credit_utilisation_pct: Optional[float] = None


class CaseInput(BaseModel):
    """Full case input — maps directly to a single entry in cases.json."""
    id: str
    name: str
    dob: str
    postcode: str
    city: str
    case_type: str
    product: str
    applicant: ApplicantInput
    bureau: BureauInput
    fraud_aml: FraudAmlInput
    open_banking: OpenBankingInput
    ai_assessed_limit_gbp: Optional[float] = None
    ai_assessed_limit_condition: Optional[str] = None


# ---------------------------------------------------------------------------
# Output schema  (what Module A returns — consumed by Module F / FastAPI)
# ---------------------------------------------------------------------------

class RiskDirection(str, Enum):
    INCREASING = "increasing"
    REDUCING = "reducing"
    AML_FRAUD = "aml_fraud"


class RiskFactor(BaseModel):
    name: str = Field(..., description="Short factor label, max 40 chars")
    weight: int = Field(..., ge=0, le=100, description="Influence weight 0–100")
    direction: RiskDirection
    explanation: str = Field(..., description="One-sentence explanation for audit log")


class AIVerdict(str, Enum):
    APPROVE = "approve"
    CONDITIONAL_APPROVE = "conditional_approve"
    DECLINE = "decline"
    REFER_SENIOR = "refer_senior"
    EDD_HOLD = "edd_hold"
    AML_HOLD = "aml_hold"


class ConfidenceLevel(str, Enum):
    HIGH = "high"    # >= 75
    MID = "mid"      # 50–74
    LOW = "low"      # < 50


class CreditMemo(BaseModel):
    """
    Structured AI credit memo — Module A output.
    Consumed by: Module F orchestrator, FastAPI response, frontend AI panel.
    """

    # Identity
    case_id: str
    applicant_name: str
    generated_at_utc: str  # ISO-8601

    # Core outputs
    verdict: AIVerdict
    confidence: int = Field(..., ge=0, le=100)
    confidence_level: ConfidenceLevel

    # Narrative (multi-paragraph, plain text — no markdown)
    narrative: str = Field(..., description="2–4 paragraph credit narrative for underwriter")

    # Recommendation
    recommended_limit_gbp: Optional[int] = Field(
        None, description="AI-assessed starting limit in GBP. None if card should not be issued."
    )
    limit_condition: Optional[str] = Field(
        None, description="Condition that must be met before limit can be applied"
    )

    # FCA CONC affordability check (computed by module, not LLM)
    fca_min_repayment_gbp: Optional[float] = None
    fca_repayment_as_pct_ndi: Optional[float] = None
    fca_conc_pass: Optional[bool] = None

    # Risk factors
    risk_factors: list[RiskFactor] = Field(..., min_length=3, max_length=10)

    # Regulatory notes (SAR, EDD, MLRO triggers)
    regulatory_notes: str = Field(
        ..., description="Specific regulatory obligations triggered by this case"
    )

    # Flags for downstream routing
    mlro_referral_required: bool = False
    edd_required: bool = False
    sar_consideration: bool = False
    senior_uw_required: bool = False

    @field_validator("confidence_level", mode="before")
    @classmethod
    def derive_confidence_level(cls, v, info):
        """Allow confidence_level to be derived from confidence if not supplied."""
        if v:
            return v
        confidence = info.data.get("confidence", 0)
        if confidence >= 75:
            return ConfidenceLevel.HIGH
        elif confidence >= 50:
            return ConfidenceLevel.MID
        return ConfidenceLevel.LOW

    @model_validator(mode="after")
    def validate_fca_fields(self) -> "CreditMemo":
        """If a limit is recommended, FCA CONC fields must be populated."""
        if self.recommended_limit_gbp is not None:
            if self.fca_conc_pass is None:
                raise ValueError(
                    "fca_conc_pass must be set when a limit is recommended"
                )
        return self


# ---------------------------------------------------------------------------
# FCA CONC affordability calculation (deterministic — not LLM)
# ---------------------------------------------------------------------------

def compute_fca_affordability(
    limit_gbp: float,
    ndi_monthly_gbp: float,
) -> tuple[float, float, bool]:
    """
    FCA CONC 5.2 stress test.

    Returns:
        (min_repayment_gbp, repayment_as_pct_ndi, passes)
    """
    min_repayment = limit_gbp * FCA_MIN_REPAYMENT_RATE
    pct_of_ndi = (min_repayment / ndi_monthly_gbp) * 100 if ndi_monthly_gbp > 0 else 999
    passes = pct_of_ndi <= FCA_MAX_NDI_REPAYMENT_PCT
    return round(min_repayment, 2), round(pct_of_ndi, 1), passes


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI credit analyst for a UK credit card issuer operating a low-to-grow near-prime strategy.

Your role is to analyse refer cases — applications that scored in the grey band (neither clean approve nor clean decline) — and produce a structured credit assessment for the underwriter.

## Your expertise covers:
- UK consumer credit risk assessment (Experian / Equifax / TransUnion bureau data)
- FCA CONC 5.2 affordability assessment obligations
- FCA Consumer Duty (PS22/9) — credit limits must be in the customer's best interest
- CIFAS and National Hunter fraud indicators
- PEP and sanctions screening under the Money Laundering Regulations 2017 (MLR 2017)
- POCA 2002 — structuring as a money laundering offence; SAR obligations
- Open banking income verification and NDI calculation
- UK near-prime credit card underwriting norms

## Key principles you always apply:
1. **Human decides** — you recommend, the underwriter approves or overrides. Your role is to prepare the fullest possible assessment.
2. **Lender sets the limit** — applicants do not request amounts. You assess and recommend the appropriate starting limit based on affordability and risk.
3. **FCA CONC 5.2** — any recommended limit must be stress-tested: minimum repayment (2% of limit) must not consume an excessive portion of NDI. Apply a 25% NDI threshold as the policy maximum.
4. **Consumer Duty** — a technically affordable limit is not always appropriate. Consider the customer's overall financial position.
5. **Three data sources** — Bureau (CRA), Fraud & AML (CIFAS/Hunter/PEP), Open Banking. Weaknesses in any source must be called out.
6. **AML/fraud triggers** — CIFAS markers, Hunter hits, structuring patterns, PEP matches, and address mismatches all require specific regulatory commentary.
7. **Regulatory honesty** — call out MLRO referral obligations, SAR considerations, and EDD requirements explicitly.

## UK terminology (always use):
- CCJs (not judgements), arrears (not delinquency), electoral roll (not voter registration)
- NDI (net disposable income), DTI ratio, credit utilisation
- CIFAS markers, National Hunter, PEP/sanctions screening
- Open banking (not UPI), all amounts in GBP £
- CRAs: Experian, Equifax, TransUnion (not CIBIL)

## Output format:
You will receive a structured JSON case object. Respond ONLY with a valid JSON object matching the schema provided. No preamble, no markdown fences, no commentary outside the JSON.
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(case: CaseInput) -> str:
    """Serialise case to a clean prompt with output schema hint."""

    output_schema = {
        "verdict": "one of: approve | conditional_approve | decline | refer_senior | edd_hold | aml_hold",
        "confidence": "integer 0–100",
        "confidence_level": "high | mid | low",
        "narrative": "2–4 paragraph credit narrative in plain text (no markdown). Cover: applicant overview, bureau findings, fraud/AML findings, open banking / affordability, recommendation rationale.",
        "recommended_limit_gbp": "integer GBP or null if card should not be issued",
        "limit_condition": "string or null — condition before limit applies",
        "risk_factors": [
            {
                "name": "short label max 40 chars",
                "weight": "0–100 influence weight",
                "direction": "increasing | reducing | aml_fraud",
                "explanation": "one sentence for audit log",
            }
        ],
        "regulatory_notes": "specific regulatory obligations triggered — SAR, EDD, MLRO, CIFAS, Consumer Duty etc.",
        "mlro_referral_required": "boolean",
        "edd_required": "boolean",
        "sar_consideration": "boolean",
        "senior_uw_required": "boolean",
    }

    return f"""Analyse the following refer case and produce a credit memo.

<case_data>
{case.model_dump_json(indent=2)}
</case_data>

<output_schema>
{json.dumps(output_schema, indent=2)}
</output_schema>

Respond with a single valid JSON object matching the output schema exactly. No other text."""


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_memo(case: CaseInput, client: Optional[anthropic.Anthropic] = None) -> CreditMemo:
    """
    Generate a structured credit memo for a refer case.

    Args:
        case: Validated CaseInput object.
        client: Optional pre-constructed Anthropic client (for dependency injection in tests).

    Returns:
        CreditMemo — fully validated Pydantic model.

    Raises:
        ValueError: If the LLM returns unparseable JSON after all retries.
        anthropic.APIError: On unrecoverable API errors.
    """
    if client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )
        client = anthropic.Anthropic(api_key=api_key)

    user_prompt = build_user_prompt(case)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(
            "Calling Claude API",
            extra={"case_id": case.id, "attempt": attempt, "model": MODEL},
        )
        t0 = time.monotonic()

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit hit on attempt {attempt}: {e}")
            last_error = e
            time.sleep(RETRY_DELAY_S * attempt)
            continue
        except anthropic.APIStatusError as e:
            logger.error(f"API error on attempt {attempt}: {e.status_code} — {e.message}")
            if e.status_code >= 500:
                last_error = e
                time.sleep(RETRY_DELAY_S)
                continue
            raise

        elapsed = time.monotonic() - t0
        logger.info(
            f"API call succeeded in {elapsed:.2f}s",
            extra={
                "case_id": case.id,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

        raw_text = response.content[0].text.strip()

        # Strip accidental markdown fences if the model adds them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        try:
            raw_dict = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed on attempt {attempt}: {e}\nRaw: {raw_text[:300]}")
            last_error = e
            time.sleep(RETRY_DELAY_S)
            continue

        # Inject fields the module computes deterministically (not LLM)
        import datetime
        raw_dict["case_id"] = case.id
        raw_dict["applicant_name"] = case.name
        raw_dict["generated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"

        # FCA CONC affordability (deterministic calculation)
        limit = raw_dict.get("recommended_limit_gbp")
        ndi = case.open_banking.ndi_monthly_gbp
        if limit and ndi:
            min_rep, pct_ndi, passes = compute_fca_affordability(float(limit), ndi)
            raw_dict["fca_min_repayment_gbp"] = min_rep
            raw_dict["fca_repayment_as_pct_ndi"] = pct_ndi
            raw_dict["fca_conc_pass"] = passes
            if not passes:
                logger.warning(
                    f"FCA CONC 5.2 fail: limit £{limit} → {pct_ndi}% of NDI £{ndi} exceeds {FCA_MAX_NDI_REPAYMENT_PCT}% threshold",
                    extra={"case_id": case.id},
                )

        # Confidence level can be derived
        confidence = raw_dict.get("confidence", 0)
        if "confidence_level" not in raw_dict:
            if confidence >= 75:
                raw_dict["confidence_level"] = "high"
            elif confidence >= 50:
                raw_dict["confidence_level"] = "mid"
            else:
                raw_dict["confidence_level"] = "low"

        try:
            memo = CreditMemo.model_validate(raw_dict)
            logger.info(
                f"Credit memo generated: {memo.verdict} | confidence {memo.confidence}%",
                extra={"case_id": case.id},
            )
            return memo
        except Exception as e:
            logger.warning(f"Pydantic validation failed on attempt {attempt}: {e}")
            last_error = e
            time.sleep(RETRY_DELAY_S)
            continue

    raise ValueError(
        f"Failed to generate valid credit memo for {case.id} after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# CLI test runner  (python -m modules.memo_generator)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cases_path = os.path.join(
        os.path.dirname(__file__), "..", "synthetic_data", "cases.json"
    )

    with open(cases_path) as f:
        data = json.load(f)

    # Default to first case; pass index as arg: python memo_generator.py 2
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    raw_case = data["cases"][idx]

    logger.info(f"Running memo generator on case index {idx}: {raw_case['id']}")

    case = CaseInput(**raw_case)
    memo = generate_memo(case)

    print("\n" + "=" * 70)
    print(f"CREDIT MEMO — {memo.case_id} — {memo.applicant_name}")
    print("=" * 70)
    print(memo.model_dump_json(indent=2))
    print("=" * 70)

    # FCA summary
    if memo.fca_conc_pass is not None:
        status = "✓ PASS" if memo.fca_conc_pass else "✗ FAIL"
        print(f"\nFCA CONC 5.2: {status}")
        print(f"  Min repayment: £{memo.fca_min_repayment_gbp}/mo")
        print(f"  As % of NDI:   {memo.fca_repayment_as_pct_ndi}%")

    # Routing flags
    flags = []
    if memo.mlro_referral_required: flags.append("MLRO referral")
    if memo.edd_required:           flags.append("EDD required")
    if memo.sar_consideration:      flags.append("SAR consideration")
    if memo.senior_uw_required:     flags.append("Senior UW required")
    if flags:
        print(f"\nRouting flags: {', '.join(flags)}")
