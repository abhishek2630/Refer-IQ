"""
Refer IQ — Synthetic Document Generator
Generates realistic UK payslips, P60s, and bank statements for the 5 test cases.
Used to test Module B (doc_parser.py).

Run: python synthetic_data/generate_documents.py
Output: synthetic_data/documents/{case_id}/
"""

import os
from datetime import date, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

W, H = A4  # 595 x 842 pts

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# ── Colour palette ──────────────────────────────────────────────────────────
PURPLE      = colors.HexColor("#4A1F6E")
PURPLE_PALE = colors.HexColor("#EDE5F5")
PURPLE_MID  = colors.HexColor("#6B3494")
AMBER       = colors.HexColor("#C47D0A")
GREEN       = colors.HexColor("#1A9E6B")
RED_C       = colors.HexColor("#B83333")
GREY_LIGHT  = colors.HexColor("#F4F2F7")
GREY_TEXT   = colors.HexColor("#5A5070")
WHITE       = colors.white
BLACK       = colors.HexColor("#1A1025")

styles = getSampleStyleSheet()


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def heading_style(size=11, color=PURPLE, bold=True):
    return ParagraphStyle(
        "h", fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size, textColor=color, leading=size * 1.3
    )


def body_style(size=9, color=BLACK):
    return ParagraphStyle(
        "b", fontName="Helvetica", fontSize=size, textColor=color, leading=size * 1.4
    )


def right_style(size=9, color=BLACK):
    return ParagraphStyle(
        "r", fontName="Helvetica", fontSize=size, textColor=color,
        leading=size * 1.4, alignment=TA_RIGHT
    )


def tbl_header_style():
    return [
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING",    (0, 0), (-1, 0), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY_LIGHT]),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("TOPPADDING",    (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#D0C8E0")),
        ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
    ]


# ── PAYSLIP GENERATOR ───────────────────────────────────────────────────────

def make_payslip(out_path, case):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )
    story = []

    # Header
    story.append(Paragraph(case["employer"], heading_style(14)))
    story.append(Paragraph("PAYSLIP", heading_style(10, PURPLE_MID)))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=6))

    # Employee + pay period info
    info_data = [
        ["Employee", case["name"],         "Pay Period",  case["pay_period"]],
        ["NI Number", case["ni"],           "Pay Date",   case["pay_date"]],
        ["Tax Code",  case["tax_code"],     "Department", case["dept"]],
        ["Payroll No",case["payroll_no"],   "Method",     "BACS transfer"],
    ]
    info_tbl = Table(info_data, colWidths=[38*mm, 58*mm, 30*mm, 46*mm])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), GREY_TEXT),
        ("TEXTCOLOR", (2, 0), (2, -1), GREY_TEXT),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, -1), GREY_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D0C8E0")),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 8))

    # Earnings table
    earn_data = [["Earnings", "Hours/Units", "Rate", "Amount"]]
    for e in case["earnings"]:
        earn_data.append([e[0], e[1], e[2], f"£{e[3]:,.2f}"])
    earn_tbl = Table(earn_data, colWidths=[80*mm, 35*mm, 35*mm, 22*mm])
    earn_tbl.setStyle(TableStyle(tbl_header_style()))
    story.append(Paragraph("Earnings", heading_style(9, GREY_TEXT, bold=False)))
    story.append(earn_tbl)
    story.append(Spacer(1, 6))

    # Deductions table
    ded_data = [["Deductions", "Amount"]]
    for d in case["deductions"]:
        ded_data.append([d[0], f"£{d[1]:,.2f}"])
    ded_tbl = Table(ded_data, colWidths=[150*mm, 22*mm])
    ded_tbl.setStyle(TableStyle(tbl_header_style()))
    story.append(Paragraph("Deductions", heading_style(9, GREY_TEXT, bold=False)))
    story.append(ded_tbl)
    story.append(Spacer(1, 8))

    # Summary box
    summary_data = [
        ["Gross pay this period",   f"£{case['gross_period']:,.2f}"],
        ["Total deductions",        f"£{case['total_deductions']:,.2f}"],
        ["NET PAY",                 f"£{case['net_pay']:,.2f}"],
        ["Gross pay YTD",           f"£{case['gross_ytd']:,.2f}"],
        ["Tax paid YTD",            f"£{case['tax_ytd']:,.2f}"],
        ["NI paid YTD",             f"£{case['ni_ytd']:,.2f}"],
    ]
    sum_tbl = Table(summary_data, colWidths=[150*mm, 22*mm])
    sum_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("FONTNAME",  (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 2), (-1, 2), 10),
        ("TEXTCOLOR", (0, 2), (-1, 2), PURPLE),
        ("BACKGROUND", (0, 2), (-1, 2), PURPLE_PALE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, GREY_LIGHT, PURPLE_PALE, GREY_LIGHT, WHITE, GREY_LIGHT]),
        ("ALIGN",     (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0C8E0")),
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#C0B8D0")))
    story.append(Paragraph(
        "This payslip is computer generated and does not require a signature. "
        "Please keep for your records. Queries: payroll@" + case["employer"].lower().replace(" ", "") + ".co.uk",
        body_style(7, GREY_TEXT)
    ))

    doc.build(story)
    print(f"  ✓ Payslip: {out_path}")


# ── BANK STATEMENT GENERATOR ─────────────────────────────────────────────────

def make_bank_statement(out_path, case):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )
    story = []

    # Bank header
    story.append(Paragraph("NatWest Bank plc", heading_style(14)))
    story.append(Paragraph("Current Account Statement", heading_style(10, PURPLE_MID)))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=6))

    # Account info
    acct_data = [
        ["Account Name",   case["name"],              "Sort Code",    case["sort_code"]],
        ["Account Number", case["account_no"],         "Statement Period", case["stmt_period"]],
        ["Statement Date", case["stmt_date"],          "Sheet",        "1 of 1"],
    ]
    acct_tbl = Table(acct_data, colWidths=[38*mm, 58*mm, 32*mm, 44*mm])
    acct_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), GREY_TEXT),
        ("TEXTCOLOR", (2, 0), (2, -1), GREY_TEXT),
        ("BACKGROUND", (0, 0), (-1, -1), GREY_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D0C8E0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(acct_tbl)
    story.append(Spacer(1, 8))

    # Opening balance
    story.append(Paragraph(
        f"Opening balance: <b>£{case['opening_balance']:,.2f}</b>",
        body_style(9)
    ))
    story.append(Spacer(1, 4))

    # Transactions table
    txn_data = [["Date", "Description", "Payments out (£)", "Payments in (£)", "Balance (£)"]]
    running = case["opening_balance"]
    for txn in case["transactions"]:
        if txn[2] != "":   # debit
            running -= txn[2]
            txn_data.append([txn[0], txn[1], f"{txn[2]:,.2f}", "", f"{running:,.2f}"])
        else:               # credit
            running += txn[3]
            txn_data.append([txn[0], txn[1], "", f"{txn[3]:,.2f}", f"{running:,.2f}"])

    txn_tbl = Table(txn_data, colWidths=[22*mm, 80*mm, 27*mm, 27*mm, 27*mm])
    style = tbl_header_style() + [
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
    ]
    txn_tbl.setStyle(TableStyle(style))
    story.append(txn_tbl)
    story.append(Spacer(1, 6))

    # Closing balance
    story.append(Paragraph(
        f"Closing balance: <b>£{running:,.2f}</b>",
        body_style(9)
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#C0B8D0")))
    story.append(Paragraph(
        "NatWest Bank plc. Registered in England & Wales (No. 929027). Registered office: 36 St Andrew Square, Edinburgh EH2 2YB. "
        "Authorised by the Prudential Regulation Authority and regulated by the Financial Conduct Authority and the Prudential Regulation Authority.",
        body_style(6, GREY_TEXT)
    ))

    doc.build(story)
    print(f"  ✓ Bank statement: {out_path}")


# ── P60 GENERATOR ───────────────────────────────────────────────────────────

def make_p60(out_path, case):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )
    story = []

    story.append(Paragraph("HM Revenue & Customs", heading_style(12, RED_C)))
    story.append(Paragraph(
        f"P60 — End of Year Certificate &nbsp;&nbsp; Tax Year {case['tax_year']}",
        heading_style(11, PURPLE)
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=RED_C, spaceAfter=8))
    story.append(Paragraph(
        "<i>To the employee: Please keep this certificate in a safe place as you may need it "
        "to complete a tax return, or claim a tax refund. Do not send it to HMRC unless asked.</i>",
        body_style(8, GREY_TEXT)
    ))
    story.append(Spacer(1, 8))

    # Employer + employee boxes
    box_data = [
        ["Employer's name", case["employer"], "Employee's name", case["name"]],
        ["Employer's PAYE ref", case["paye_ref"], "NI number", case["ni"]],
        ["Employer's address", case["employer_addr"], "Tax code at 5 April", case["tax_code"]],
    ]
    box_tbl = Table(box_data, colWidths=[40*mm, 50*mm, 40*mm, 42*mm])
    box_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), GREY_TEXT),
        ("TEXTCOLOR", (2, 0), (2, -1), GREY_TEXT),
        ("BACKGROUND", (0, 0), (-1, -1), GREY_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0C8E0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(box_tbl)
    story.append(Spacer(1, 8))

    # P60 earnings table
    p60_data = [
        ["", "In this employment", "In previous employment(s)", "Total for year"],
        ["Pay",          f"£{case['gross_annual']:,.2f}", "£0.00", f"£{case['gross_annual']:,.2f}"],
        ["Tax deducted", f"£{case['tax_annual']:,.2f}",  "£0.00", f"£{case['tax_annual']:,.2f}"],
        ["", "", "", ""],
        ["National Insurance contributions", "", "", ""],
        ["NI category letter", "A", "", ""],
        ["Earnings at the Lower Earnings Limit (LEL) or above", f"£{case['gross_annual']:,.2f}", "", ""],
        ["Employee's NI contributions",  f"£{case['ni_annual']:,.2f}", "", ""],
        ["Employer's NI contributions",  f"£{case['employer_ni']:,.2f}", "", ""],
    ]
    p60_tbl = Table(p60_data, colWidths=[80*mm, 36*mm, 36*mm, 20*mm])
    p60_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0C8E0")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Paragraph("Pay and Income Tax details", heading_style(9, GREY_TEXT, bold=False)))
    story.append(p60_tbl)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#C0B8D0")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This form shows your total pay and the amount of tax and National Insurance deducted during the tax year shown above.",
        body_style(7, GREY_TEXT)
    ))

    doc.build(story)
    print(f"  ✓ P60: {out_path}")


# ── CASE DEFINITIONS ─────────────────────────────────────────────────────────

CASES_DOCS = [

    # REF-2025-00141 — James Mellor — AML / structuring
    {
        "case_id": "REF-2025-00141",
        "payslip": {
            "name": "James Mellor", "employer": "Logistics Co Ltd",
            "ni": "JM 74 25 83 A", "tax_code": "1257L",
            "dept": "Operations", "payroll_no": "LCL-0419",
            "pay_period": "March 2025", "pay_date": "28 Mar 2025",
            "earnings": [
                ("Basic salary", "160 hrs", "£14.84", 2375.0),
                ("Overtime", "12 hrs", "£22.26", 267.12),
            ],
            "deductions": [
                ("Income Tax (PAYE)", 312.80),
                ("National Insurance (Employee)", 198.40),
                ("Pension (Workplace — 3%)", 71.22),
            ],
            "gross_period": 2375.0 + 267.12,
            "total_deductions": 312.80 + 198.40 + 71.22,
            "net_pay": 2059.70,
            "gross_ytd": 26812.50,
            "tax_ytd": 3487.50,
            "ni_ytd": 2185.00,
        },
        "bank_stmt": {
            "name": "James Mellor", "sort_code": "40-27-14",
            "account_no": "81204763",
            "stmt_period": "01 Jan 2025 — 31 Mar 2025",
            "stmt_date": "31 Mar 2025",
            "opening_balance": 1240.50,
            "transactions": [
                ("03 Jan", "Cash deposit — branch", 9750.0, ""),    # structuring
                ("05 Jan", "Direct Debit — Sky TV", 49.99, ""),
                ("10 Jan", "Faster Payment — rent", 750.00, ""),
                ("28 Jan", "Salary — Logistics Co Ltd", "", 2059.70),
                ("02 Feb", "Cash deposit — branch", 8900.0, ""),    # structuring
                ("05 Feb", "Direct Debit — Sky TV", 49.99, ""),
                ("10 Feb", "Faster Payment — rent", 750.00, ""),
                ("14 Feb", "Card — Tesco grocery", 187.40, ""),
                ("28 Feb", "Salary — Logistics Co Ltd", "", 2059.70),
                ("04 Mar", "Cash deposit — branch", 9450.0, ""),    # structuring
                ("05 Mar", "Direct Debit — Sky TV", 49.99, ""),
                ("10 Mar", "Faster Payment — rent", 750.00, ""),
                ("21 Mar", "Card — Barclaycard min payment", 45.00, ""),
                ("21 Mar", "Card — Amazon min payment", 30.00, ""),
                ("28 Mar", "Salary — Logistics Co Ltd", "", 2059.70),
            ],
        },
        "p60": {
            "name": "James Mellor", "employer": "Logistics Co Ltd",
            "ni": "JM 74 25 83 A", "tax_code": "1257L",
            "paye_ref": "120/LCL1047", "tax_year": "2023–24",
            "employer_addr": "Unit 7 Wharfside Park\nLeeds LS10 1AB",
            "gross_annual": 28500, "tax_annual": 3750,
            "ni_annual": 2340, "employer_ni": 3105,
        },
    },

    # REF-2025-00138 — Priya Sharma — Bureau / thin file
    {
        "case_id": "REF-2025-00138",
        "payslip": {
            "name": "Priya Sharma", "employer": "Barts Health NHS Trust",
            "ni": "PS 42 17 96 C", "tax_code": "1257L",
            "dept": "Radiology — Band 4", "payroll_no": "NHS-P3748",
            "pay_period": "March 2025", "pay_date": "27 Mar 2025",
            "earnings": [
                ("Basic salary (Band 4 pt 4)", "150 hrs", "£12.74", 1911.0),
                ("Unsocial hours supplement", "18 hrs", "£3.18", 57.24),
            ],
            "deductions": [
                ("Income Tax (PAYE)", 148.60),
                ("National Insurance (Employee)", 127.80),
                ("NHS Pension (5.2%)", 101.24),
            ],
            "gross_period": 1911.0 + 57.24,
            "total_deductions": 148.60 + 127.80 + 101.24,
            "net_pay": 1590.60,
            "gross_ytd": 22104.0,
            "tax_ytd": 1678.0,
            "ni_ytd": 1411.20,
        },
        "bank_stmt": {
            "name": "Priya Sharma", "sort_code": "20-14-33",
            "account_no": "54903817",
            "stmt_period": "01 Jan 2025 — 31 Mar 2025",
            "stmt_date": "31 Mar 2025",
            "opening_balance": 2145.80,
            "transactions": [
                ("01 Jan", "Direct Debit — EDF Energy", 87.00, ""),
                ("05 Jan", "Faster Payment — rent", 950.00, ""),
                ("15 Jan", "Card — Sainsburys grocery", 143.20, ""),
                ("27 Jan", "Salary — Barts Health NHS Trust", "", 1590.60),
                ("01 Feb", "Direct Debit — EDF Energy", 87.00, ""),
                ("05 Feb", "Faster Payment — rent", 950.00, ""),
                ("14 Feb", "Card — Deliveroo", 34.50, ""),
                ("18 Feb", "Card — Waterstones", 19.99, ""),
                ("27 Feb", "Salary — Barts Health NHS Trust", "", 1590.60),
                ("01 Mar", "Direct Debit — EDF Energy", 87.00, ""),
                ("05 Mar", "Faster Payment — rent", 950.00, ""),
                ("10 Mar", "Card — store card min payment", 15.00, ""),
                ("20 Mar", "Card — Sainsburys grocery", 156.40, ""),
                ("27 Mar", "Salary — Barts Health NHS Trust", "", 1590.60),
            ],
        },
        "p60": {
            "name": "Priya Sharma", "employer": "Barts Health NHS Trust",
            "ni": "PS 42 17 96 C", "tax_code": "1257L",
            "paye_ref": "475/BART0291", "tax_year": "2023–24",
            "employer_addr": "80 Newark Street\nLondon E1 2ES",
            "gross_annual": 19200, "tax_annual": 1428,
            "ni_annual": 1382.40, "employer_ni": 1843.20,
        },
    },

    # REF-2025-00133 — Marcus Webb — Fraud
    {
        "case_id": "REF-2025-00133",
        "payslip": {
            "name": "Marcus Webb", "employer": "Marcus Webb Contracting",
            "ni": "MW 61 33 74 A", "tax_code": "1257L",
            "dept": "Director / Self-employed", "payroll_no": "MWC-DIR-001",
            "pay_period": "March 2025", "pay_date": "31 Mar 2025",
            "earnings": [
                ("Director's remuneration", "—", "—", 1083.33),
                ("Dividend payment", "—", "—", 1750.00),
            ],
            "deductions": [
                ("Income Tax (PAYE on salary)", 0.00),
                ("National Insurance (Class 1)", 0.00),
                ("National Insurance (Class 2 — SE)", 3.45),
            ],
            "gross_period": 1083.33 + 1750.00,
            "total_deductions": 3.45,
            "net_pay": 2829.88,
            "gross_ytd": 9749.97,
            "tax_ytd": 0.00,
            "ni_ytd": 31.05,
        },
        "bank_stmt": {
            "name": "Marcus Webb", "sort_code": "30-91-47",
            "account_no": "22713048",
            "stmt_period": "01 Jan 2025 — 31 Mar 2025",
            "stmt_date": "31 Mar 2025",
            "opening_balance": 874.20,
            "transactions": [
                ("06 Jan", "Transfer from MWC Ltd — remuneration", "", 1083.33),
                ("06 Jan", "Transfer from MWC Ltd — dividend", "", 1750.00),
                ("10 Jan", "Faster Payment — rent (B12 address)", 1100.00, ""),
                ("15 Jan", "Card — Amazon", 134.50, ""),
                ("20 Jan", "Gambling — Bet365 deposit", 200.00, ""),
                ("22 Jan", "Gambling — Bet365 withdrawal", "", 155.00),
                ("06 Feb", "Transfer from MWC Ltd — remuneration", "", 1083.33),
                ("06 Feb", "Transfer from MWC Ltd — dividend", "", 1750.00),
                ("10 Feb", "Faster Payment — rent (B12 address)", 1100.00, ""),
                ("14 Feb", "Card — Costa Coffee", 4.80, ""),
                ("05 Mar", "Transfer from MWC Ltd — remuneration", "", 1083.33),
                ("05 Mar", "Transfer from MWC Ltd — dividend", "", 1750.00),
                ("12 Mar", "Faster Payment — rent (B15 address)",  1100.00, ""),  # address change
                ("20 Mar", "Gambling — Bet365 deposit", 50.00, ""),
            ],
        },
        "p60": {
            "name": "Marcus Webb", "employer": "Marcus Webb Contracting",
            "ni": "MW 61 33 74 A", "tax_code": "1257L",
            "paye_ref": "220/MWC4417", "tax_year": "2023–24",
            "employer_addr": "14 Colmore Row\nBirmingham B3 2QD",
            "gross_annual": 13000, "tax_annual": 0,
            "ni_annual": 41.40, "employer_ni": 0,
        },
    },

    # REF-2025-00129 — Aoife Byrne — AML / PEP
    {
        "case_id": "REF-2025-00129",
        "payslip": {
            "name": "Aoife Byrne", "employer": "Aoife Byrne Consulting Ltd",
            "ni": "AB 88 49 12 D", "tax_code": "1257L",
            "dept": "Director / Consultant", "payroll_no": "ABC-DIR-001",
            "pay_period": "March 2025", "pay_date": "31 Mar 2025",
            "earnings": [
                ("Director's salary", "—", "—", 1041.67),
                ("Consulting fee — March", "—", "—", 2400.00),
            ],
            "deductions": [
                ("Income Tax (PAYE on salary)", 0.00),
                ("National Insurance (Employee)", 0.00),
                ("Self-assessment provision (est.)", 325.00),
            ],
            "gross_period": 1041.67 + 2400.00,
            "total_deductions": 325.00,
            "net_pay": 3116.67,
            "gross_ytd": 31250.01,
            "tax_ytd": 0.00,
            "ni_ytd": 0.00,
        },
        "bank_stmt": {
            "name": "Aoife Byrne", "sort_code": "60-11-29",
            "account_no": "38475920",
            "stmt_period": "01 Jan 2025 — 31 Mar 2025",
            "stmt_date": "31 Mar 2025",
            "opening_balance": 5820.00,
            "transactions": [
                ("03 Jan", "BACS — client fee Innovate UK Q4", "", 4800.00),
                ("05 Jan", "Faster Payment — rent M4 1HQ", 1200.00, ""),
                ("08 Jan", "Direct Debit — Vodafone", 42.00, ""),
                ("15 Jan", "Card — Waitrose", 189.60, ""),
                ("31 Jan", "Director salary — Aoife Byrne Consulting", "", 1041.67),
                ("04 Feb", "BACS — client fee TechNorth Ltd", "", 2400.00),
                ("05 Feb", "Faster Payment — rent M4 1HQ", 1200.00, ""),
                ("12 Feb", "Card — Amex min payment", 78.00, ""),
                ("28 Feb", "Director salary — Aoife Byrne Consulting", "", 1041.67),
                ("06 Mar", "BACS — client fee Innovate UK Feb", "", 2400.00),
                ("10 Mar", "Faster Payment — rent M4 1HQ", 1200.00, ""),
                ("15 Mar", "Card — M&S Food", 214.30, ""),
                ("31 Mar", "Director salary — Aoife Byrne Consulting", "", 1041.67),
            ],
        },
        "p60": {
            "name": "Aoife Byrne", "employer": "Aoife Byrne Consulting Ltd",
            "ni": "AB 88 49 12 D", "tax_code": "1257L",
            "paye_ref": "319/ABC2284", "tax_year": "2023–24",
            "employer_addr": "Office 5, Piccadilly House\nManchester M1 2NX",
            "gross_annual": 12500, "tax_annual": 0,
            "ni_annual": 0, "employer_ni": 0,
        },
    },

    # REF-2025-00121 — David Okafor — Bureau / grey band
    {
        "case_id": "REF-2025-00121",
        "payslip": {
            "name": "David Okafor", "employer": "Transport for London",
            "ni": "DO 55 81 27 B", "tax_code": "1257L",
            "dept": "Network Operations", "payroll_no": "TfL-NOps-2847",
            "pay_period": "March 2025", "pay_date": "25 Mar 2025",
            "earnings": [
                ("Basic salary", "160 hrs", "£17.15", 2744.0),
                ("Shift allowance", "—", "—", 210.0),
            ],
            "deductions": [
                ("Income Tax (PAYE)", 448.80),
                ("National Insurance (Employee)", 285.20),
                ("TfL Pension (7%)", 213.08),
            ],
            "gross_period": 2744.0 + 210.0,
            "total_deductions": 448.80 + 285.20 + 213.08,
            "net_pay": 2006.92,
            "gross_ytd": 32340.0,
            "tax_ytd": 4716.0,
            "ni_ytd": 2988.0,
        },
        "bank_stmt": {
            "name": "David Okafor", "sort_code": "20-98-52",
            "account_no": "70314892",
            "stmt_period": "01 Jan 2025 — 31 Mar 2025",
            "stmt_date": "31 Mar 2025",
            "opening_balance": 1876.40,
            "transactions": [
                ("02 Jan", "Direct Debit — Thames Water", 38.00, ""),
                ("05 Jan", "Faster Payment — rent SE15", 900.00, ""),
                ("10 Jan", "Card — Barclaycard min payment", 28.00, ""),
                ("10 Jan", "Card — Santander min payment", 22.00, ""),
                ("10 Jan", "Card — Amazon min payment", 15.00, ""),
                ("25 Jan", "Salary — Transport for London", "", 2006.92),
                ("02 Feb", "Direct Debit — Thames Water", 38.00, ""),
                ("05 Feb", "Faster Payment — rent SE15", 900.00, ""),
                ("10 Feb", "Card — Barclaycard min payment", 28.00, ""),
                ("10 Feb", "Card — Santander min payment", 22.00, ""),
                ("10 Feb", "Card — Amazon min payment", 15.00, ""),
                ("20 Feb", "Card — Tesco grocery", 198.50, ""),
                ("25 Feb", "Salary — Transport for London", "", 2006.92),
                ("05 Mar", "Faster Payment — rent SE15", 900.00, ""),
                ("10 Mar", "Card — Barclaycard min payment", 28.00, ""),
                ("10 Mar", "Card — Santander min payment", 22.00, ""),
                ("10 Mar", "Card — Amazon min payment", 15.00, ""),
                ("18 Mar", "Card — Lidl grocery", 167.30, ""),
                ("25 Mar", "Salary — Transport for London", "", 2006.92),
            ],
        },
        "p60": {
            "name": "David Okafor", "employer": "Transport for London",
            "ni": "DO 55 81 27 B", "tax_code": "1257L",
            "paye_ref": "480/TFL0022", "tax_year": "2023–24",
            "employer_addr": "55 Broadway\nLondon SW1H 0BD",
            "gross_annual": 32000, "tax_annual": 4356,
            "ni_annual": 2844, "employer_ni": 3984,
        },
    },
]


def generate_all():
    print(f"\nGenerating synthetic documents in: {DOCS_DIR}\n")
    for case in CASES_DOCS:
        cid = case["case_id"]
        out_dir = os.path.join(DOCS_DIR, cid)
        make_dir(out_dir)
        print(f"Case {cid}:")
        make_payslip(os.path.join(out_dir, "payslip_mar2025.pdf"), case["payslip"])
        make_bank_statement(os.path.join(out_dir, "bank_statement_q1_2025.pdf"), case["bank_stmt"])
        make_p60(os.path.join(out_dir, "p60_2023_24.pdf"), case["p60"])
        print()

    print("Done. Documents created for all 5 cases.")


if __name__ == "__main__":
    generate_all()
