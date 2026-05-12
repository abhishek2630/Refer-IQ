"""
Refer IQ — Module B: Document Parser
=====================================
Takes a synthetic payslip, P60, or bank statement PDF and extracts structured
financial fields using the Claude vision API (base64 PDF).

Input:  PDF file path (payslip / P60 / bank statement)
Output: ParsedDocument Pydantic model — see schema below

Usage:
    from modules.doc_parser import parse_document, DocumentType
    result = parse_document("synthetic_data/documents/REF-2025-00141/payslip_mar2025.pdf")
    print(result.model_dump_json(indent=2))

Run tests directly:
    python modules/doc_parser.py
"""

import os
import base64
import json
import re
from enum import Enum
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ── Output schema ──────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    PAYSLIP       = "payslip"
    P60           = "p60"
    BANK_STATEMENT = "bank_statement"
    UNKNOWN       = "unknown"


class TransactionRecord(BaseModel):
    """Single bank statement transaction."""
    date: str
    description: str
    debit: Optional[float] = None   # payment out
    credit: Optional[float] = None  # payment in
    balance: Optional[float] = None


class ParsedDocument(BaseModel):
    """
    Structured output from the document parser.
    This is the schema expected by Module C (affordability engine)
    and Module F (orchestrator).
    """
    # Metadata
    doc_type:          DocumentType
    source_file:       str
    parse_confidence:  float = Field(ge=0.0, le=1.0, description="0-1 parser confidence")

    # Core income fields (all modules need these)
    employee_name:     Optional[str]   = None
    employer_name:     Optional[str]   = None
    ni_number:         Optional[str]   = None
    tax_code:          Optional[str]   = None
    pay_period:        Optional[str]   = None
    pay_date:          Optional[str]   = None
    tax_year:          Optional[str]   = None  # P60 only

    # Payslip / P60 monetary fields
    gross_monthly:     Optional[float] = None  # current period gross
    net_monthly:       Optional[float] = None  # current period net (take-home)
    gross_annual:      Optional[float] = None  # YTD or P60 annual figure
    tax_paid_period:   Optional[float] = None
    ni_paid_period:    Optional[float] = None
    pension_period:    Optional[float] = None
    tax_paid_annual:   Optional[float] = None  # YTD / P60
    ni_paid_annual:    Optional[float] = None  # YTD / P60
    pay_frequency:     Optional[str]   = None  # "monthly" | "weekly" | "4-weekly"

    # Bank statement fields
    account_number:    Optional[str]   = None
    sort_code:         Optional[str]   = None
    statement_period:  Optional[str]   = None
    opening_balance:   Optional[float] = None
    closing_balance:   Optional[float] = None
    total_credits:     Optional[float] = None
    total_debits:      Optional[float] = None
    avg_monthly_credit: Optional[float] = None
    transactions:      Optional[list[TransactionRecord]] = None

    # Flags extracted from document content
    adverse_flags:     list[str] = Field(default_factory=list)
    # e.g. ["cash_deposits_detected", "gambling_merchant_detected", "irregular_salary"]

    # Raw extraction notes from the LLM
    extraction_notes:  Optional[str] = None

    @field_validator("gross_monthly", "net_monthly", "gross_annual", mode="before")
    @classmethod
    def clean_currency(cls, v):
        """Strip £ signs and commas if LLM returns a string."""
        if isinstance(v, str):
            v = v.replace("£", "").replace(",", "").strip()
            return float(v) if v else None
        return v


# ── Prompt templates ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a specialist financial document extraction system for a UK credit card underwriting platform.
Your role is to extract precise structured data from UK financial documents — payslips, P60s, and bank statements.

Rules:
- Always respond with a single valid JSON object matching the schema provided.
- Do NOT include markdown backticks or preamble — raw JSON only.
- Monetary values must be plain floats (no £ signs, no commas). E.g. 2059.70 not "£2,059.70".
- If a field is not present in the document, return null for that field.
- For bank statements, extract ALL transactions as a list.
- Identify adverse flags:
  * "cash_deposits_detected" — any cash deposit transactions present
  * "gambling_merchant_detected" — any gambling-related transactions (Bet365, Ladbrokes, etc.)
  * "irregular_salary" — salary credits are not consistent month-on-month
  * "payday_loan_detected" — any payday lender transactions
  * "rent_payment_address_change" — rent payments reference different addresses across the statement
  * "multiple_sub_threshold_cash" — 2+ cash deposits each below £10,000 in the period
- parse_confidence: your confidence in the extraction quality (0.0–1.0).
- extraction_notes: brief notes on anything unusual or uncertain about the extraction."""

PAYSLIP_USER_PROMPT = """Extract all fields from this UK payslip. Return JSON matching this schema exactly:

{
  "doc_type": "payslip",
  "source_file": "",
  "parse_confidence": 0.0,
  "employee_name": null,
  "employer_name": null,
  "ni_number": null,
  "tax_code": null,
  "pay_period": null,
  "pay_date": null,
  "tax_year": null,
  "gross_monthly": null,
  "net_monthly": null,
  "gross_annual": null,
  "tax_paid_period": null,
  "ni_paid_period": null,
  "pension_period": null,
  "tax_paid_annual": null,
  "ni_paid_annual": null,
  "pay_frequency": null,
  "account_number": null,
  "sort_code": null,
  "statement_period": null,
  "opening_balance": null,
  "closing_balance": null,
  "total_credits": null,
  "total_debits": null,
  "avg_monthly_credit": null,
  "transactions": null,
  "adverse_flags": [],
  "extraction_notes": null
}"""

BANK_STMT_USER_PROMPT = """Extract all fields from this UK bank statement. Include every transaction.
Return JSON matching this schema exactly:

{
  "doc_type": "bank_statement",
  "source_file": "",
  "parse_confidence": 0.0,
  "employee_name": null,
  "employer_name": null,
  "ni_number": null,
  "tax_code": null,
  "pay_period": null,
  "pay_date": null,
  "tax_year": null,
  "gross_monthly": null,
  "net_monthly": null,
  "gross_annual": null,
  "tax_paid_period": null,
  "ni_paid_period": null,
  "pension_period": null,
  "tax_paid_annual": null,
  "ni_paid_annual": null,
  "account_number": null,
  "sort_code": null,
  "statement_period": null,
  "opening_balance": null,
  "closing_balance": null,
  "total_credits": null,
  "total_debits": null,
  "avg_monthly_credit": null,
  "transactions": [
    {"date": null, "description": null, "debit": null, "credit": null, "balance": null}
  ],
  "adverse_flags": [],
  "extraction_notes": null
}"""

P60_USER_PROMPT = """Extract all fields from this UK P60 (End of Year Certificate). Return JSON matching this schema exactly:

{
  "doc_type": "p60",
  "source_file": "",
  "parse_confidence": 0.0,
  "employee_name": null,
  "employer_name": null,
  "ni_number": null,
  "tax_code": null,
  "pay_period": null,
  "pay_date": null,
  "tax_year": null,
  "gross_monthly": null,
  "net_monthly": null,
  "gross_annual": null,
  "tax_paid_period": null,
  "ni_paid_period": null,
  "pension_period": null,
  "tax_paid_annual": null,
  "ni_paid_annual": null,
  "account_number": null,
  "sort_code": null,
  "statement_period": null,
  "opening_balance": null,
  "closing_balance": null,
  "total_credits": null,
  "total_debits": null,
  "avg_monthly_credit": null,
  "transactions": null,
  "adverse_flags": [],
  "extraction_notes": null
}"""


def _detect_doc_type(filename: str) -> tuple[DocumentType, str]:
    """
    Detect document type from filename and return (type, user_prompt).
    Falls back to UNKNOWN / payslip prompt if uncertain.
    """
    name = filename.lower()
    if "payslip" in name:
        return DocumentType.PAYSLIP, PAYSLIP_USER_PROMPT
    if "bank_statement" in name or "statement" in name:
        return DocumentType.BANK_STATEMENT, BANK_STMT_USER_PROMPT
    if "p60" in name:
        return DocumentType.P60, P60_USER_PROMPT
    return DocumentType.UNKNOWN, PAYSLIP_USER_PROMPT


def _pdf_to_base64(pdf_path: str) -> str:
    """Read a PDF file and return it as a base64-encoded string."""
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _clean_json_response(text: str) -> str:
    """Strip any accidental markdown fences the model might add."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _compute_bank_statement_totals(parsed: ParsedDocument) -> ParsedDocument:
    """
    Post-process bank statement: compute total_credits, total_debits,
    avg_monthly_credit from the transactions list if the LLM didn't fill them.
    """
    if parsed.doc_type != DocumentType.BANK_STATEMENT:
        return parsed
    if not parsed.transactions:
        return parsed

    txns = parsed.transactions
    total_credits = sum(t.credit for t in txns if t.credit is not None)
    total_debits  = sum(t.debit  for t in txns if t.debit  is not None)

    if parsed.total_credits is None:
        parsed.total_credits = round(total_credits, 2)
    if parsed.total_debits is None:
        parsed.total_debits = round(total_debits, 2)

    # Estimate avg monthly credit: assume statement period covers ~3 months
    # unless statement_period gives us something to parse
    period_months = 3
    if parsed.statement_period:
        months_match = re.search(r"(\d+)\s*month", parsed.statement_period, re.IGNORECASE)
        if months_match:
            period_months = int(months_match.group(1))

    if parsed.avg_monthly_credit is None and period_months > 0:
        parsed.avg_monthly_credit = round(total_credits / period_months, 2)

    # Detect adverse flags if LLM missed them
    descs = [t.description.lower() for t in txns if t.description]
    credits = [t for t in txns if t.credit is not None]

    cash_deposits = [d for d in descs if "cash deposit" in d or "cash dep" in d]
    if cash_deposits and "cash_deposits_detected" not in parsed.adverse_flags:
        parsed.adverse_flags.append("cash_deposits_detected")
        # Sub-threshold structuring check (cash deposits < £10k each)
        cash_amounts = [
            t.credit for t in txns
            if t.credit is not None and t.description and
            ("cash deposit" in t.description.lower() or "cash dep" in t.description.lower())
        ]
        if len([a for a in cash_amounts if a < 10000]) >= 2:
            parsed.adverse_flags.append("multiple_sub_threshold_cash")

    gambling_keywords = ["bet365", "betfair", "william hill", "ladbrokes", "paddy power",
                         "sky bet", "flutter", "betway", "coral", "888sport", "gambling"]
    if any(kw in desc for desc in descs for kw in gambling_keywords):
        if "gambling_merchant_detected" not in parsed.adverse_flags:
            parsed.adverse_flags.append("gambling_merchant_detected")

    payday_keywords = ["wonga", "quickquid", "sunny", "myjar", "lending stream",
                       "payday", "cashfloat", "peachy"]
    if any(kw in desc for desc in descs for kw in payday_keywords):
        if "payday_loan_detected" not in parsed.adverse_flags:
            parsed.adverse_flags.append("payday_loan_detected")

    return parsed


def parse_document(
    pdf_path: str,
    api_key: Optional[str] = None,
) -> ParsedDocument:
    """
    Parse a financial document PDF and return a structured ParsedDocument.

    Args:
        pdf_path:  Path to the PDF file to parse.
        api_key:   Anthropic API key (defaults to ANTHROPIC_API_KEY env var).

    Returns:
        ParsedDocument with extracted fields.

    Raises:
        FileNotFoundError: if pdf_path doesn't exist.
        ValueError: if the API response cannot be parsed.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {pdf_path}")

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — add it to .env")

    doc_type, user_prompt = _detect_doc_type(path.name)
    pdf_b64 = _pdf_to_base64(str(path))

    client = anthropic.Anthropic(api_key=key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text
    clean = _clean_json_response(raw)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON for {path.name}.\n"
            f"Error: {e}\nRaw response:\n{raw[:500]}"
        ) from e

    # Inject source file path (LLM can't know this)
    data["source_file"] = str(path)

    parsed = ParsedDocument(**data)

    # Post-process bank statements
    parsed = _compute_bank_statement_totals(parsed)

    return parsed


def parse_case_documents(
    case_id: str,
    docs_dir: str = "synthetic_data/documents",
    api_key: Optional[str] = None,
) -> dict[str, ParsedDocument]:
    """
    Parse all available documents for a given case.

    Args:
        case_id:  e.g. "REF-2025-00141"
        docs_dir: root documents directory
        api_key:  optional override for ANTHROPIC_API_KEY

    Returns:
        Dict mapping doc type name to ParsedDocument:
        {"payslip": ..., "bank_statement": ..., "p60": ...}
    """
    case_dir = Path(docs_dir) / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"No document directory found for case {case_id}")

    results: dict[str, ParsedDocument] = {}
    pdf_files = sorted(case_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {case_dir}")

    for pdf_path in pdf_files:
        doc_type, _ = _detect_doc_type(pdf_path.name)
        print(f"  Parsing {pdf_path.name} → {doc_type.value}…")
        parsed = parse_document(str(pdf_path), api_key=api_key)
        results[doc_type.value] = parsed

    return results


# ── CLI test runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Allow override: python modules/doc_parser.py REF-2025-00138
    test_case = sys.argv[1] if len(sys.argv) > 1 else "REF-2025-00141"

    print("=" * 70)
    print(f"Module B — Document Parser — Test run: {test_case}")
    print("=" * 70)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    docs = parse_case_documents(
        case_id=test_case,
        docs_dir="synthetic_data/documents",
    )

    for dtype, parsed in docs.items():
        print(f"\n{'─'*60}")
        print(f"  Document type: {parsed.doc_type.value.upper()}")
        print(f"  Source:        {Path(parsed.source_file).name}")
        print(f"  Confidence:    {parsed.parse_confidence:.0%}")
        print(f"{'─'*60}")

        if parsed.doc_type == DocumentType.PAYSLIP:
            print(f"  Employee:      {parsed.employee_name}")
            print(f"  Employer:      {parsed.employer_name}")
            print(f"  Pay period:    {parsed.pay_period}")
            print(f"  Pay date:      {parsed.pay_date}")
            print(f"  Gross (period):£{parsed.gross_monthly:,.2f}" if parsed.gross_monthly else "  Gross: —")
            print(f"  Net (period):  £{parsed.net_monthly:,.2f}"  if parsed.net_monthly  else "  Net: —")
            print(f"  Gross YTD:     £{parsed.gross_annual:,.2f}" if parsed.gross_annual  else "  Gross YTD: —")
            print(f"  Tax YTD:       £{parsed.tax_paid_annual:,.2f}" if parsed.tax_paid_annual else "  Tax YTD: —")
            print(f"  NI YTD:        £{parsed.ni_paid_annual:,.2f}"  if parsed.ni_paid_annual  else "  NI YTD: —")
            print(f"  Tax code:      {parsed.tax_code}")
            print(f"  NI number:     {parsed.ni_number}")

        elif parsed.doc_type == DocumentType.BANK_STATEMENT:
            print(f"  Account name:  {parsed.employee_name}")
            print(f"  Sort code:     {parsed.sort_code}")
            print(f"  Account no:    {parsed.account_number}")
            print(f"  Period:        {parsed.statement_period}")
            print(f"  Opening bal:   £{parsed.opening_balance:,.2f}" if parsed.opening_balance else "  Opening: —")
            print(f"  Closing bal:   £{parsed.closing_balance:,.2f}" if parsed.closing_balance else "  Closing: —")
            print(f"  Total credits: £{parsed.total_credits:,.2f}"   if parsed.total_credits   else "  Credits: —")
            print(f"  Total debits:  £{parsed.total_debits:,.2f}"    if parsed.total_debits    else "  Debits: —")
            print(f"  Avg mo. credit:£{parsed.avg_monthly_credit:,.2f}" if parsed.avg_monthly_credit else "  Avg credit: —")
            if parsed.transactions:
                print(f"  Transactions:  {len(parsed.transactions)} rows extracted")

        elif parsed.doc_type == DocumentType.P60:
            print(f"  Employee:      {parsed.employee_name}")
            print(f"  Employer:      {parsed.employer_name}")
            print(f"  Tax year:      {parsed.tax_year}")
            print(f"  Gross annual:  £{parsed.gross_annual:,.2f}" if parsed.gross_annual else "  Gross: —")
            print(f"  Tax annual:    £{parsed.tax_paid_annual:,.2f}" if parsed.tax_paid_annual else "  Tax: —")
            print(f"  NI annual:     £{parsed.ni_paid_annual:,.2f}"  if parsed.ni_paid_annual  else "  NI: —")
            print(f"  NI number:     {parsed.ni_number}")

        if parsed.adverse_flags:
            print(f"\n  ⚠ Adverse flags detected: {', '.join(parsed.adverse_flags)}")
        else:
            print(f"\n  ✓ No adverse flags")

        if parsed.extraction_notes:
            print(f"\n  Notes: {parsed.extraction_notes}")

    print("\n" + "=" * 70)
    print("Module B — PASS ✓" if docs else "Module B — FAIL ✗")
    print("=" * 70)
