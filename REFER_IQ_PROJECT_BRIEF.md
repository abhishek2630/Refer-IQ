# Refer IQ — Project Brief & Technical Handover

> **Pin this file in the Project.** It is the authoritative context document for all module development sessions. Start every session by referencing this brief.

---

## What we are building

**Refer IQ** is an AI-powered refer queue platform for a UK credit card issuer operating a **low-to-grow near-prime strategy**. When a credit card application scores in the grey band (neither a clean approve nor a clean decline), the originations system flags it as a refer case. Refer IQ processes the case automatically across three data sources and surfaces a pre-built AI assessment to the underwriter — reducing decision time from ~50 minutes to under 10, and enabling consistent, FCA-compliant decisions at scale.

The platform is generic (no client name in the codebase) but designed specifically for UK credit card acquisition. All nomenclature, regulation references, and data fields are UK-specific.

---

## Current state — what is already built

### Frontend demo (`index.html`)
A fully working single-file HTML/CSS/JS demo hosted on GitHub Pages or Netlify. Two tabs:

**Tab 1 — Underwriter workbench**
- Live refer queue (5 pre-loaded cases)
- Three-panel layout: queue list / case detail / AI recommendation
- Per-case: applicant overview, three-source tabbed data view (Bureau / Fraud & AML / Open Banking), AI narrative, AI recommendation with confidence score and risk factors, similar past cases panel
- Case Q&A chat (currently static keyword-matched answers)
- Decision bar: Issue card (AI limit), Override limit, Decline, Refer up
- Alert banners for AML and fraud cases
- Stats bar: queue count, avg decision time, AI concordance, STP rate, daily approvals

**Tab 2 — Solution architecture**
- Animated 12-step flow diagram showing end-to-end data movement
- Canvas-based animation with play/pause/step/speed controls
- Covers: LOS trigger → data pull → AI pipeline → LLM synthesis → UW decision → feedback loop

### Design system
- Font: DM Sans + DM Serif Display (Google Fonts)
- Primary colour: `#4A1F6E` (deep purple)
- Accent: `#FF6B35` (orange)
- Semantic: green `#1A9E6B` / amber `#C47D0A` / red `#B83333`
- No external CSS framework — pure custom CSS with CSS variables

### Five synthetic cases already defined

| Case ID | Name | Refer type | AI verdict | Key signals |
|---|---|---|---|---|
| REF-2025-00141 | James Mellor | AML — cash structuring | Conditional issue £1,500 (post-AML) | Sub-£10k deposits × 3 months, util 81%, NDI £347 |
| REF-2025-00138 | Priya Sharma | Bureau — thin file | Issue card £2,000 | Score 592, NHS employed, NDI £490, clean all sources |
| REF-2025-00133 | Marcus Webb | Fraud — CIFAS Cat.6 + Hunter | Refer to senior UW | CIFAS Cat.6, 2 Hunter hits, address mismatch, CCJ |
| REF-2025-00129 | Aoife Byrne | AML — PEP match | Hold for EDD | PEP screening match, strong affordability, SA302 income |
| REF-2025-00121 | David Okafor | Bureau — grey band | Issue card £2,000 | Score 577, TfL employed, util 67%, clean fraud/AML |

### Case data schema (JavaScript object per case)
```js
{
  id, name, initials, dob, postcode, time, caseType,
  tags: [{t, c}],           // queue tag text + CSS class
  aiDot, aiLabel,           // queue AI pre-recommendation
  alertType, alertText,     // "aml" | "fraud" | null
  app: { emp, employer, tenure, gross, net, product, aiLimit },
  bureau: { rows: [[label, value, colorClass]], flags: [{t, c}] },
  fraud:  { rows: [...], flags: [...] },
  ob:     { rows: [...], flags: [...] },
  narrative,                // multi-paragraph string
  ai: {
    verdict, icon, iCls, sub,
    conf, cCls,             // confidence 0-100, "high"|"mid"|"low"
    note, nCls,
    factors: [{n, w, fc, dc, dch}],  // name, weight%, fill-class, dir-class, char
    similar: [{m, o, c}],   // meta, outcome, css class
    hints: [],              // quick-ask chip labels
    qa: { keyword: answer } // case-specific Q&A dictionary
  },
  approveLabel              // text for primary decision button
}
```

---

## Three data sources (the core of the platform)

Every refer case is assessed across exactly three sources:

| Source | What it covers | UK-specific detail |
|---|---|---|
| **Bureau (CRA)** | Credit score, CCJs, defaults, arrears, hard searches, electoral roll | Experian / Equifax / TransUnion — not CIBIL |
| **Fraud & AML** | CIFAS markers, National Hunter hits, PEP/sanctions screening, identity verification, address consistency | CIFAS is UK's shared fraud prevention database; National Hunter is the shared application fraud DB |
| **Open Banking** | Income verification (90-day feed), NDI calculation, spending patterns, structuring detection, gambling/payday activity | FCA regulated; TrueLayer / Plaid / Experian OB are the main providers in UK |

---

## Module build plan — agreed sequence

Build one module at a time, each producing working code tested against synthetic data, wired to the Claude API.

### Module A — AI credit memo generator *(build first)*
**What it does:** Takes a structured case object (from the five synthetic cases) → calls Claude API → returns a drafted credit narrative + AI recommendation + key risk factors.

**Inputs:** Structured JSON with bureau, fraud/AML, and open banking fields.
**Outputs:** `{ narrative: string, verdict: string, confidence: number, factors: [], suggestedLimit: number, regulatoryNotes: string }`
**API:** Claude claude-sonnet-4-20250514, system prompt encodes UK credit policy + FCA CONC rules.
**Files to create:** `modules/memo_generator.py` + `synthetic_data/cases.json`

### Module B — Document parser
**What it does:** Takes a synthetic payslip or bank statement PDF → extracts structured fields (gross pay, net pay, employer, pay date, account credits/debits).
**Inputs:** PDF file (synthetic documents to be created).
**Outputs:** `{ gross_monthly: float, net_monthly: float, employer: string, pay_frequency: string, ob_credits: [], ob_debits: [] }`
**API:** Claude vision API (base64 PDF/image).
**Files to create:** `modules/doc_parser.py` + `synthetic_data/documents/`

### Module C — Affordability engine
**What it does:** Pure calculation module. Takes income and commitment data → computes NDI, DTI, stress-tested minimum repayment capacity, and recommended credit limit band.
**Inputs:** `{ gross_monthly, net_monthly, existing_commitments: [], rent_estimate }`
**Outputs:** `{ ndi: float, dti: float, max_limit_fca: int, recommended_limit: int, stress_result: string }`
**Regulation:** FCA CONC 5.2 affordability thresholds.
**Files to create:** `modules/affordability.py`

### Module D — Fraud & AML signal aggregator
**What it does:** Takes raw fraud/AML signals → scores the case on a fraud risk scale → flags specific concerns (structuring, CIFAS, Hunter, PEP, address mismatch) with explanations.
**Inputs:** Synthetic CIFAS/Hunter/PEP JSON responses.
**Outputs:** `{ fraud_risk_score: int, flags: [], mlro_referral_required: bool, edd_required: bool, explanation: string }`
**Files to create:** `modules/fraud_aml.py` + `synthetic_data/fraud_responses/`

### Module E — RAG pipeline (policy + past cases)
**What it does:** Embeds credit policy documents and past decided cases into a vector store → retrieves relevant policy rules and similar cases for each new refer.
**Inputs:** Policy docs (FCA CONC excerpts, internal credit policy), past case outcomes.
**Outputs:** `{ relevant_policy: [], similar_cases: [], policy_citations: [] }`
**Stack:** ChromaDB (local vector store) + Claude embeddings or sentence-transformers.
**Files to create:** `modules/rag_pipeline.py` + `data/policy_docs/` + `data/past_cases/`

### Module F — Case orchestrator
**What it does:** Wires A–E together. Receives a case trigger → runs all modules in parallel → assembles the final case output object → returns to frontend API.
**Files to create:** `modules/orchestrator.py` + `api/main.py` (FastAPI)

### Module G — Frontend wiring
**What it does:** Replaces the static `CASES` array and hardcoded Q&A dictionary in `index.html` with live API calls to the FastAPI backend.
**Files to create:** Updated `index.html` with fetch() calls + WebSocket for real-time processing status.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Claude claude-sonnet-4-20250514 via Anthropic Python SDK |
| Backend | Python 3.11 + FastAPI |
| Vector store | ChromaDB (local, no external service needed) |
| Document parsing | Claude vision API (base64) |
| Embeddings | `sentence-transformers` or Claude embeddings endpoint |
| Frontend | Vanilla HTML/CSS/JS (existing `index.html`) |
| Env management | `python-dotenv` — API key in `.env`, never hardcoded |
| Dev environment | macOS, conda `base` env, pyenv Python 3.11 |

---

## Project folder structure (target)

```
refer-iq/
├── index.html                  ← existing frontend (two tabs)
├── README.md
├── .env                        ← ANTHROPIC_API_KEY (gitignored)
├── .gitignore
├── requirements.txt
│
├── modules/
│   ├── memo_generator.py       ← Module A
│   ├── doc_parser.py           ← Module B
│   ├── affordability.py        ← Module C
│   ├── fraud_aml.py            ← Module D
│   ├── rag_pipeline.py         ← Module E
│   └── orchestrator.py         ← Module F
│
├── api/
│   └── main.py                 ← FastAPI app (Module F)
│
├── synthetic_data/
│   ├── cases.json              ← 5 structured test cases
│   ├── documents/              ← synthetic payslips, P60s, bank statements
│   ├── fraud_responses/        ← synthetic CIFAS/Hunter API responses
│   └── ob_feeds/               ← synthetic open banking transaction feeds
│
└── data/
    ├── policy_docs/            ← FCA CONC excerpts, internal credit policy
    └── past_cases/             ← historical decided cases for RAG
```

---

## UK regulatory context (always apply)

- **FCA CONC 5.2** — affordability assessment obligation. Credit limit must be stress-tested against minimum repayment as % of NDI.
- **FCA Consumer Duty (PS22/9)** — credit limit must be in the customer's best interest, not just technically affordable.
- **POCA 2002** — structuring (sub-threshold deposits) is a criminal offence. SAR submission to NCA required on suspicion.
- **MLR 2017** — EDD required for all PEP relationships. Senior management sign-off before account opening.
- **UK GDPR** — all personal data handling must be documented and compliant.
- **PRA / SR 26-2** — model risk management obligations for AI models used in credit decisions.
- **Equality Act 2010** — decisions must be monitored for bias across protected characteristics.

**UK-specific terminology always used:**
- CRAs: Experian, Equifax, TransUnion (not CIBIL)
- CCJs (not judgements), arrears (not delinquency), electoral roll (not voter registration)
- NDI (net disposable income), DTI ratio, credit utilisation
- CIFAS markers, National Hunter, PEP/sanctions screening
- Open banking (not UPI), salary in GBP
- Card issuer sets the limit — applicants do not request a specific amount

---

## How to start each session

1. Read this brief.
2. State which module you are working on and what the goal is for this session.
3. Reference the synthetic case data (cases.json once created, or the JS CASES array in index.html).
4. Build, test against synthetic data, verify output schema matches what the next module expects.
5. End each session by confirming what was built, what the output looks like, and what Module comes next.

---

## Key decisions already made

- **No client name** in any code, filename, or comment. The platform is generic.
- **Lender sets the limit** — no "requested amount" field anywhere. AI assesses and recommends the starting limit.
- **Three data sources** are always Bureau + Fraud/AML (CIFAS/Hunter/PEP) + Open Banking. This is the core differentiation from manual review.
- **Human always decides** — AI recommends, underwriter approves/overrides. Override is always logged.
- **Synthetic data first** — build and validate every module with synthetic data before any real integration.
- **One module per session** — do not skip ahead. Each module must have working, tested code before the next begins.

---

*Brief last updated: May 2026. Prepared from the design and build conversation in Claude.ai chat.*
