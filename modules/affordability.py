"""
Refer IQ — Module C: Affordability Engine
==========================================
Pure calculation module — no API calls, no external dependencies beyond Pydantic.

Takes verified income and existing financial commitments and computes:
  - NDI  (Net Disposable Income)
  - DTI  (Debt-to-Income ratio)
  - Stress-tested minimum repayment capacity
  - Recommended credit limit band (FCA CONC 5.2 compliant)

Regulation applied:
  - FCA CONC 5.2  — affordability assessment obligation
  - FCA Consumer Duty (PS22/9) — limit must be in customer's best interest
  - FCA CONC 5.2.8 — stress test: minimum repayment must not exceed 10% NDI
                      for near-prime / thin-file applicants (internal policy threshold)

Input:  AffordabilityInput Pydantic model
Output: AffordabilityResult Pydantic model

Usage:
    from modules.affordability import assess_affordability, AffordabilityInput
    result = assess_affordability(AffordabilityInput(
        gross_monthly=2375.0,
        net_monthly=1920.0,
        existing_commitments=[...],
        estimated_rent=750.0,
    ))
    print(result.model_dump_json(indent=2))

Run tests directly:
    python modules/affordability.py
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ── Constants — FCA CONC 5.2 + internal credit policy ────────────────────────

# Minimum repayment rate assumed for all revolving credit (2% of balance or £25, whichever greater)
MIN_REPAYMENT_RATE = 0.02

# FCA CONC stress threshold: minimum repayment on new card must not exceed
# this % of post-commitment NDI for near-prime applicants
CONC_STRESS_THRESHOLD_PCT = 0.10   # 10%

# Consumer Duty soft threshold: where utilisation is high (>60%) we apply a
# tighter stress threshold to avoid loading a customer with more revolving debt
CONSUMER_DUTY_STRESS_PCT  = 0.08   # 8% when existing utilisation > 60%

# DTI policy limits
DTI_AMBER_THRESHOLD = 0.40   # 40% — flag for review
DTI_RED_THRESHOLD   = 0.55   # 55% — decline territory

# Credit limit bands (£) — lender sets the limit, not the applicant
LIMIT_BANDS = [500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 5000]

# Essential spend estimate if open banking data not available (% of net income)
ESSENTIAL_SPEND_FALLBACK_PCT = 0.20  # 20% of net monthly as fallback


# ── Input schema ──────────────────────────────────────────────────────────────

class CommitmentType(str, Enum):
    CREDIT_CARD    = "credit_card"
    PERSONAL_LOAN  = "personal_loan"
    MORTGAGE       = "mortgage"
    HIRE_PURCHASE  = "hire_purchase"
    OVERDRAFT      = "overdraft"
    OTHER          = "other"


class ExistingCommitment(BaseModel):
    """A single existing financial commitment."""
    commitment_type:    CommitmentType
    description:        str                          # e.g. "Barclaycard", "Car finance"
    outstanding_balance: Optional[float] = None      # current balance
    monthly_payment:    float                        # actual or minimum monthly payment
    credit_limit:       Optional[float] = None       # for revolving credit
    utilisation_pct:    Optional[float] = None       # 0-100


class AffordabilityInput(BaseModel):
    """
    Input to the affordability engine.
    All monetary values in GBP, monthly unless stated.
    """
    # Core income (required)
    gross_monthly:          float = Field(gt=0, description="Gross monthly income in £")
    net_monthly:            float = Field(gt=0, description="Net (take-home) monthly income in £")

    # Existing commitments
    existing_commitments:   list[ExistingCommitment] = Field(default_factory=list)

    # Housing costs
    estimated_rent:         Optional[float] = None   # monthly rent from OB or declared
    has_mortgage:           bool = False

    # Essential spend (from open banking if available, else use fallback)
    essential_spend_monthly: Optional[float] = None  # groceries, utilities, transport

    # Applicant context (affects stress threshold)
    credit_utilisation_pct: Optional[float] = None   # overall existing utilisation 0-100
    is_self_employed:       bool = False
    is_thin_file:           bool = False

    # Near-prime policy cap — lender strategy: low-to-grow starting limits
    # Set to True for near-prime / grey-band applicants (default for this platform)
    apply_near_prime_cap:   bool = True
    near_prime_max_limit:   int  = 2000   # starting limit cap for grey-band strategy

    # Product being assessed
    product_min_repayment_rate: float = MIN_REPAYMENT_RATE

    @model_validator(mode="after")
    def net_cannot_exceed_gross(self):
        if self.net_monthly > self.gross_monthly:
            raise ValueError("net_monthly cannot exceed gross_monthly")
        return self


# ── Output schema ─────────────────────────────────────────────────────────────

class StressResult(str, Enum):
    PASS_COMFORTABLE   = "pass_comfortable"    # repayment < 8% NDI
    PASS_WITHIN_POLICY = "pass_within_policy"  # repayment 8-10% NDI
    FAIL_EXCEEDS_CONC  = "fail_exceeds_conc"   # repayment > 10% NDI — cannot lend at this limit
    FAIL_NEGATIVE_NDI  = "fail_negative_ndi"   # NDI is negative after commitments


class RiskBand(str, Enum):
    GREEN  = "green"   # strong affordability, clean profile
    AMBER  = "amber"   # marginal — lend at lower band
    RED    = "red"     # cannot lend — policy breach


class AffordabilityResult(BaseModel):
    """
    Output of the affordability engine.
    This is the schema consumed by Module A (memo generator) and Module F (orchestrator).
    """
    # Core affordability metrics
    ndi:                    float   = Field(description="Net Disposable Income £/month after all commitments")
    ndi_post_new_card:      float   = Field(description="NDI after adding new card minimum repayment")
    dti:                    float   = Field(description="Debt-to-Income ratio 0-1 (existing commitments / gross)")
    dti_with_new_card:      float   = Field(description="DTI including new card at recommended limit")
    total_monthly_commitments: float

    # Stress test
    stress_result:          StressResult
    stress_threshold_used:  float   = Field(description="CONC stress threshold applied (0.08 or 0.10)")
    new_card_min_repayment: float   = Field(description="Minimum repayment on new card at recommended limit £/month")
    repayment_pct_of_ndi:   float   = Field(description="New card repayment as % of current NDI")

    # Limit recommendation
    recommended_limit:      int     = Field(description="AI-recommended starting credit limit £")
    max_limit_fca:          int     = Field(description="Maximum limit that passes CONC stress test £")
    risk_band:              RiskBand

    # Supporting breakdown
    gross_monthly:          float
    net_monthly:            float
    estimated_rent:         float
    essential_spend:        float
    existing_commitment_total: float
    existing_revolving_total:  float   = Field(description="Total existing revolving credit balance")
    existing_utilisation_pct:  Optional[float]

    # Flags and notes
    policy_flags:           list[str] = Field(default_factory=list)
    # e.g. ["dti_exceeds_amber", "high_utilisation", "self_employed_income_variable"]
    conc_compliant:         bool
    consumer_duty_note:     Optional[str] = None
    calculation_notes:      str


# ── Core calculation logic ────────────────────────────────────────────────────

def _calculate_ndi(
    net_monthly: float,
    rent: float,
    essential_spend: float,
    commitment_payments: float,
) -> float:
    """NDI = net income − rent − essential spend − existing commitment payments."""
    return round(net_monthly - rent - essential_spend - commitment_payments, 2)


def _calculate_dti(
    total_monthly_payments: float,
    gross_monthly: float,
) -> float:
    """DTI = total monthly minimum payments / gross monthly income."""
    if gross_monthly == 0:
        return 0.0
    return round(total_monthly_payments / gross_monthly, 4)


def _find_max_conc_limit(
    ndi: float,
    stress_threshold: float,
    min_repayment_rate: float,
) -> int:
    """
    Find the highest limit band whose minimum repayment does not exceed
    stress_threshold % of NDI (FCA CONC 5.2).

    min repayment = max(limit * rate, £25)
    We iterate from highest to lowest and return the first passing band.
    """
    if ndi <= 0:
        return 0

    max_affordable_repayment = ndi * stress_threshold

    for limit in reversed(LIMIT_BANDS):
        min_repayment = max(limit * min_repayment_rate, 25.0)
        if min_repayment <= max_affordable_repayment:
            return limit

    return 0  # no band is affordable


def _select_recommended_limit(
    max_fca_limit: int,
    ndi: float,
    dti: float,
    utilisation_pct: Optional[float],
    is_thin_file: bool,
    is_self_employed: bool,
) -> tuple[int, list[str]]:
    """
    Apply Consumer Duty adjustments to step down from the max FCA-compliant limit.
    Returns (recommended_limit, list_of_adjustment_reasons).

    Rules:
    1. Start at max_fca_limit
    2. If existing utilisation > 60% → step down one band (Consumer Duty)
    3. If DTI > 40% → step down one band
    4. If self-employed → step down one band (income volatility)
    5. If thin file → no step-down (thin file is a data gap, not a risk signal)
    6. Never go below £500
    """
    if max_fca_limit == 0:
        return 0, ["ndi_insufficient_no_limit_possible"]

    adjustments = []
    idx = LIMIT_BANDS.index(max_fca_limit) if max_fca_limit in LIMIT_BANDS else len(LIMIT_BANDS) - 1

    if utilisation_pct is not None and utilisation_pct > 60:
        if idx > 0:
            idx -= 1
            adjustments.append(f"consumer_duty_step_down: utilisation {utilisation_pct:.0f}% > 60%")

    if dti > DTI_AMBER_THRESHOLD:
        if idx > 0:
            idx -= 1
            adjustments.append(f"dti_step_down: DTI {dti:.0%} > 40% policy threshold")

    if is_self_employed:
        if idx > 0:
            idx -= 1
            adjustments.append("self_employed_step_down: variable income risk")

    recommended = LIMIT_BANDS[max(idx, 0)]
    recommended = max(recommended, 500)

    return recommended, adjustments


def assess_affordability(inp: AffordabilityInput) -> AffordabilityResult:
    """
    Run the full affordability assessment and return an AffordabilityResult.

    This is the primary entry point for Module C.
    """
    notes = []
    policy_flags = []

    # ── 1. Housing costs ──────────────────────────────────────────────────────
    rent = inp.estimated_rent or 0.0
    if inp.estimated_rent is None:
        notes.append("Rent not provided — using £0. Underwriter should verify housing costs.")
        policy_flags.append("rent_not_verified")

    # ── 2. Essential spend ────────────────────────────────────────────────────
    if inp.essential_spend_monthly is not None:
        essential_spend = inp.essential_spend_monthly
        notes.append(f"Essential spend from open banking: £{essential_spend:,.2f}/mo")
    else:
        essential_spend = round(inp.net_monthly * ESSENTIAL_SPEND_FALLBACK_PCT, 2)
        notes.append(
            f"Essential spend estimated at {ESSENTIAL_SPEND_FALLBACK_PCT:.0%} of net income "
            f"(£{essential_spend:,.2f}/mo) — open banking data not supplied."
        )
        policy_flags.append("essential_spend_estimated")

    # ── 3. Existing commitments ───────────────────────────────────────────────
    commitment_payments = sum(c.monthly_payment for c in inp.existing_commitments)
    revolving_balance   = sum(
        c.outstanding_balance or 0.0
        for c in inp.existing_commitments
        if c.commitment_type in (CommitmentType.CREDIT_CARD, CommitmentType.OVERDRAFT)
    )

    # Existing utilisation (use provided value, or compute from commitments if possible)
    existing_util = inp.credit_utilisation_pct
    if existing_util is None:
        revolving_limits = sum(
            c.credit_limit or 0.0
            for c in inp.existing_commitments
            if c.credit_limit is not None
        )
        if revolving_limits > 0:
            existing_util = round((revolving_balance / revolving_limits) * 100, 1)

    # ── 4. NDI ────────────────────────────────────────────────────────────────
    ndi = _calculate_ndi(inp.net_monthly, rent, essential_spend, commitment_payments)

    if ndi <= 0:
        notes.append(f"NDI is negative (£{ndi:,.2f}) — applicant cannot service existing commitments from income.")
        policy_flags.append("negative_ndi")

    # ── 5. DTI ────────────────────────────────────────────────────────────────
    dti = _calculate_dti(commitment_payments, inp.gross_monthly)

    if dti > DTI_RED_THRESHOLD:
        policy_flags.append(f"dti_exceeds_red_threshold: {dti:.0%}")
    elif dti > DTI_AMBER_THRESHOLD:
        policy_flags.append(f"dti_exceeds_amber_threshold: {dti:.0%}")

    if existing_util is not None and existing_util > 60:
        policy_flags.append(f"high_existing_utilisation: {existing_util:.0f}%")

    if inp.is_self_employed:
        policy_flags.append("self_employed_income_variable")

    if inp.is_thin_file:
        policy_flags.append("thin_file_data_gap_not_risk_signal")
        notes.append("Thin file flag noted — this is a data gap, not adverse history. No limit step-down applied.")

    # ── 6. Stress threshold selection (CONC 5.2 + Consumer Duty) ─────────────
    stress_threshold = (
        CONSUMER_DUTY_STRESS_PCT
        if (existing_util is not None and existing_util > 60)
        else CONC_STRESS_THRESHOLD_PCT
    )
    notes.append(
        f"Stress threshold: {stress_threshold:.0%} NDI "
        f"({'Consumer Duty tighter threshold — util >60%' if stress_threshold == CONSUMER_DUTY_STRESS_PCT else 'FCA CONC 5.2 standard threshold'})"
    )

    # ── 7. Maximum FCA-compliant limit ────────────────────────────────────────
    max_fca_limit = _find_max_conc_limit(ndi, stress_threshold, inp.product_min_repayment_rate)
    notes.append(f"Max FCA CONC 5.2 compliant limit: £{max_fca_limit:,}")

    # ── 8. Recommended limit (Consumer Duty adjustments) ─────────────────────
    recommended_limit, adjustments = _select_recommended_limit(
        max_fca_limit, ndi, dti,
        existing_util,
        inp.is_thin_file,
        inp.is_self_employed,
    )
    if adjustments:
        notes.extend(adjustments)

    # Near-prime policy cap -- low-to-grow strategy: cap starting limits for
    # grey-band applicants regardless of what pure affordability supports.
    # Module F (orchestrator) can override this for prime-band referrals.
    if inp.apply_near_prime_cap and recommended_limit > inp.near_prime_max_limit:
        notes.append(
            f"Near-prime policy cap applied: limit reduced from £{recommended_limit:,} "
            f"to £{inp.near_prime_max_limit:,} (low-to-grow strategy -- grey band applicant)"
        )
        policy_flags.append(f"near_prime_cap_applied: £{inp.near_prime_max_limit:,}")
        recommended_limit = inp.near_prime_max_limit

    notes.append(f"Recommended starting limit: £{recommended_limit:,}")

    # ── 9. Stress test result at recommended limit ────────────────────────────
    new_card_min_repayment = max(recommended_limit * inp.product_min_repayment_rate, 25.0) if recommended_limit > 0 else 0.0
    repayment_pct_of_ndi   = (new_card_min_repayment / ndi) if ndi > 0 else 999.0
    ndi_post_new_card       = round(ndi - new_card_min_repayment, 2)
    dti_with_new_card       = _calculate_dti(commitment_payments + new_card_min_repayment, inp.gross_monthly)

    if ndi <= 0:
        stress_result = StressResult.FAIL_NEGATIVE_NDI
    elif recommended_limit == 0:
        stress_result = StressResult.FAIL_EXCEEDS_CONC
    elif repayment_pct_of_ndi <= CONSUMER_DUTY_STRESS_PCT:
        stress_result = StressResult.PASS_COMFORTABLE
    elif repayment_pct_of_ndi <= CONC_STRESS_THRESHOLD_PCT:
        stress_result = StressResult.PASS_WITHIN_POLICY
    else:
        stress_result = StressResult.FAIL_EXCEEDS_CONC

    conc_compliant = stress_result in (StressResult.PASS_COMFORTABLE, StressResult.PASS_WITHIN_POLICY)

    # ── 10. Risk band ─────────────────────────────────────────────────────────
    if not conc_compliant or ndi <= 0:
        risk_band = RiskBand.RED
    elif dti > DTI_AMBER_THRESHOLD or (existing_util is not None and existing_util > 60):
        risk_band = RiskBand.AMBER
    else:
        risk_band = RiskBand.GREEN

    # ── 11. Consumer Duty narrative note ─────────────────────────────────────
    consumer_duty_note = None
    if adjustments:
        consumer_duty_note = (
            f"Recommended limit (£{recommended_limit:,}) is below the maximum FCA-compliant "
            f"limit (£{max_fca_limit:,}) due to Consumer Duty considerations: "
            + "; ".join(adjustments) + "."
        )

    return AffordabilityResult(
        ndi=ndi,
        ndi_post_new_card=ndi_post_new_card,
        dti=dti,
        dti_with_new_card=dti_with_new_card,
        total_monthly_commitments=round(commitment_payments, 2),
        stress_result=stress_result,
        stress_threshold_used=stress_threshold,
        new_card_min_repayment=round(new_card_min_repayment, 2),
        repayment_pct_of_ndi=round(repayment_pct_of_ndi, 4),
        recommended_limit=recommended_limit,
        max_limit_fca=max_fca_limit,
        risk_band=risk_band,
        gross_monthly=inp.gross_monthly,
        net_monthly=inp.net_monthly,
        estimated_rent=rent,
        essential_spend=essential_spend,
        existing_commitment_total=round(commitment_payments, 2),
        existing_revolving_total=round(revolving_balance, 2),
        existing_utilisation_pct=existing_util,
        policy_flags=policy_flags,
        conc_compliant=conc_compliant,
        consumer_duty_note=consumer_duty_note,
        calculation_notes=" | ".join(notes),
    )


# ── CLI test runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pydantic import ValidationError

    print("=" * 70)
    print("Module C — Affordability Engine — Test run: all 5 synthetic cases")
    print("=" * 70)

    TEST_CASES = [
        {
            "label": "REF-2025-00141 — James Mellor (AML / high util)",
            "input": AffordabilityInput(
                gross_monthly=2375.0,
                net_monthly=1920.0,
                estimated_rent=750.0,
                essential_spend_monthly=320.0,
                credit_utilisation_pct=81.0,
                existing_commitments=[
                    ExistingCommitment(
                        commitment_type=CommitmentType.CREDIT_CARD,
                        description="Barclaycard",
                        monthly_payment=45.0,
                        outstanding_balance=1800.0,
                        credit_limit=2000.0,
                        utilisation_pct=90,
                    ),
                    ExistingCommitment(
                        commitment_type=CommitmentType.CREDIT_CARD,
                        description="Amazon Card",
                        monthly_payment=30.0,
                        outstanding_balance=750.0,
                        credit_limit=1000.0,
                        utilisation_pct=75,
                    ),
                ],
            ),
            "expected_limit": 2000,   # Module F applies AML cap -> £1,500
            "expected_band": RiskBand.AMBER,
        },
        {
            "label": "REF-2025-00138 — Priya Sharma (thin file / clean)",
            "input": AffordabilityInput(
                gross_monthly=1968.24,
                net_monthly=1590.60,
                estimated_rent=950.0,
                essential_spend_monthly=230.0,
                credit_utilisation_pct=22.0,
                is_thin_file=True,
                existing_commitments=[
                    ExistingCommitment(
                        commitment_type=CommitmentType.CREDIT_CARD,
                        description="Store card",
                        monthly_payment=15.0,
                        outstanding_balance=180.0,
                        credit_limit=500.0,
                        utilisation_pct=36,
                    ),
                ],
            ),
            "expected_limit": 1500,   # NDI supports £1,500; underwriter may note thin-file uplift rationale
            "expected_band": RiskBand.GREEN,
        },
        {
            "label": "REF-2025-00133 — Marcus Webb (fraud / self-employed)",
            "input": AffordabilityInput(
                gross_monthly=2833.33,
                net_monthly=2829.88,
                estimated_rent=1100.0,
                essential_spend_monthly=400.0,
                credit_utilisation_pct=None,
                is_self_employed=True,
                existing_commitments=[],
            ),
            "expected_limit": 2000,   # Module F applies fraud block -> do-not-issue
            "expected_band": RiskBand.GREEN,
        },
        {
            "label": "REF-2025-00129 — Aoife Byrne (PEP / self-employed / strong NDI)",
            "input": AffordabilityInput(
                gross_monthly=3441.67,
                net_monthly=2791.67,
                estimated_rent=1200.0,
                essential_spend_monthly=350.0,
                credit_utilisation_pct=8.0,
                is_self_employed=True,
                existing_commitments=[
                    ExistingCommitment(
                        commitment_type=CommitmentType.CREDIT_CARD,
                        description="Amex",
                        monthly_payment=78.0,
                        outstanding_balance=620.0,
                        credit_limit=5000.0,
                        utilisation_pct=12,
                    ),
                ],
            ),
            "expected_limit": 2000,   # Module F applies PEP block -> £1,500 post-EDD
            "expected_band": RiskBand.GREEN,
        },
        {
            "label": "REF-2025-00121 — David Okafor (grey band / high util)",
            "input": AffordabilityInput(
                gross_monthly=2954.0,
                net_monthly=2006.92,
                estimated_rent=900.0,
                essential_spend_monthly=280.0,
                credit_utilisation_pct=67.0,
                existing_commitments=[
                    ExistingCommitment(
                        commitment_type=CommitmentType.CREDIT_CARD,
                        description="Barclaycard",
                        monthly_payment=28.0,
                        outstanding_balance=1120.0,
                        credit_limit=1500.0,
                        utilisation_pct=75,
                    ),
                    ExistingCommitment(
                        commitment_type=CommitmentType.CREDIT_CARD,
                        description="Santander",
                        monthly_payment=22.0,
                        outstanding_balance=880.0,
                        credit_limit=1500.0,
                        utilisation_pct=59,
                    ),
                    ExistingCommitment(
                        commitment_type=CommitmentType.CREDIT_CARD,
                        description="Amazon",
                        monthly_payment=15.0,
                        outstanding_balance=450.0,
                        credit_limit=1000.0,
                        utilisation_pct=45,
                    ),
                ],
            ),
            "expected_limit": 2000,
            "expected_band": RiskBand.AMBER,
        },
    ]

    all_pass = True

    for tc in TEST_CASES:
        print(f"\n{'─'*68}")
        print(f"  {tc['label']}")
        print(f"{'─'*68}")

        result = assess_affordability(tc["input"])

        print(f"  NDI:                  £{result.ndi:,.2f} / mo")
        print(f"  NDI post new card:    £{result.ndi_post_new_card:,.2f} / mo")
        print(f"  DTI (existing):       {result.dti:.1%}")
        print(f"  DTI (with new card):  {result.dti_with_new_card:.1%}")
        print(f"  New card min payment: £{result.new_card_min_repayment:.2f} / mo")
        print(f"  Repayment % of NDI:   {result.repayment_pct_of_ndi:.1%}")
        print(f"  Stress result:        {result.stress_result.value}")
        print(f"  Max FCA CONC limit:   £{result.max_limit_fca:,}")
        print(f"  Recommended limit:    £{result.recommended_limit:,}")
        print(f"  Risk band:            {result.risk_band.value.upper()}")
        print(f"  CONC compliant:       {'✓ Yes' if result.conc_compliant else '✗ No'}")

        if result.policy_flags:
            print(f"  Policy flags:         {', '.join(result.policy_flags)}")
        if result.consumer_duty_note:
            print(f"  Consumer Duty note:   {result.consumer_duty_note}")

        limit_ok = result.recommended_limit == tc["expected_limit"]
        band_ok  = result.risk_band == tc["expected_band"]
        case_ok  = limit_ok and band_ok
        all_pass = all_pass and case_ok

        print(f"\n  Limit: {'✓' if limit_ok else '✗'} Got £{result.recommended_limit:,} | Expected £{tc['expected_limit']:,}")
        print(f"  Band:  {'✓' if band_ok  else '✗'} Got {result.risk_band.value} | Expected {tc['expected_band'].value}")

    print(f"\n{'='*70}")
    print(f"Module C — {'PASS ✓' if all_pass else 'FAIL ✗'} — {sum(1 for tc in TEST_CASES if assess_affordability(tc['input']).recommended_limit == tc['expected_limit'])}/{len(TEST_CASES)} limits correct")
    print("=" * 70)
