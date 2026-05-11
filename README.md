# Refer IQ — AI-Powered Refer Queue Platform

AI-assisted underwriting for UK credit card originations. Built for a **low-to-grow near-prime strategy** — processes refer cases automatically across three data sources and surfaces a pre-built AI assessment to the underwriter.

**Decision time: ~50 min manual → under 10 min with AI assistance.**

---

## What it does

When a credit card application scores in the grey band (neither clean approve nor clean decline), the originations system flags it as a refer case. Refer IQ:

1. Pulls data from three sources in parallel — Bureau (CRA), Fraud & AML (CIFAS/Hunter/PEP), Open Banking
2. Runs structured AI analysis via Claude API
3. Generates a credit narrative, verdict, confidence score, and risk factors
4. Surfaces everything to the underwriter in a purpose-built workbench UI
5. Logs all decisions and AI overrides to audit trail

Human always decides. AI recommends.

---

## Project structure

```
refer-iq/
├── index.html                  ← Underwriter workbench frontend (two-tab UI)
├── requirements.txt
├── .gitignore
├── REFER_IQ_PROJECT_BRIEF.md   ← Full technical specification
│
├── modules/
│   ├── memo_generator.py       ← Module A: AI credit memo generator ✅
│   ├── doc_parser.py           ← Module B: Document parser (coming)
│   ├── affordability.py        ← Module C: Affordability engine (coming)
│   ├── fraud_aml.py            ← Module D: Fraud & AML aggregator (coming)
│   ├── rag_pipeline.py         ← Module E: RAG pipeline (coming)
│   └── orchestrator.py         ← Module F: Case orchestrator (coming)
│
├── api/
│   └── main.py                 ← FastAPI app / Module F (coming)
│
├── synthetic_data/
│   ├── cases.json              ← 5 structured test cases ✅
│   ├── documents/              ← Synthetic payslips, P60s (coming)
│   ├── fraud_responses/        ← Synthetic CIFAS/Hunter responses (coming)
│   └── ob_feeds/               ← Synthetic open banking feeds (coming)
│
└── data/
    ├── policy_docs/            ← FCA CONC excerpts, credit policy (coming)
    └── past_cases/             ← Historical decided cases for RAG (coming)
```

---

## Module build status

| Module | Description | Status |
|--------|-------------|--------|
| A — AI credit memo generator | Case JSON → Claude API → `CreditMemo` Pydantic model | ✅ Built & tested |
| B — Document parser | PDF payslips/bank statements → structured income fields | 🔜 Next |
| C — Affordability engine | NDI, DTI, stress test, FCA CONC 5.2 limits | 🔜 |
| D — Fraud & AML aggregator | CIFAS/Hunter/PEP signals → fraud risk score + flags | 🔜 |
| E — RAG pipeline | Policy docs + past cases → ChromaDB vector store | 🔜 |
| F — Case orchestrator | Wires A–E → FastAPI endpoint | 🔜 |
| G — Frontend wiring | Replace static data with live API calls | 🔜 |

---

## Setup

```bash
# Python 3.11 recommended
pip install -r requirements.txt

# Add your Anthropic API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# Test Module A against synthetic cases
python -m modules.memo_generator 0   # James Mellor — AML structuring
python -m modules.memo_generator 1   # Priya Sharma — thin file
python -m modules.memo_generator 2   # Marcus Webb — fraud indicators
python -m modules.memo_generator 3   # Aoife Byrne — PEP match
python -m modules.memo_generator 4   # David Okafor — grey band bureau
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| LLM | Claude claude-sonnet-4-20250514 via Anthropic Python SDK |
| Backend | Python 3.11 + FastAPI |
| Data validation | Pydantic v2 |
| Vector store | ChromaDB (local) |
| Document parsing | Claude vision API |
| Embeddings | sentence-transformers |
| Frontend | Vanilla HTML/CSS/JS |
| Env | python-dotenv |

---

## UK regulatory context

- **FCA CONC 5.2** — affordability obligation; stress-tested on every limit recommendation
- **FCA Consumer Duty (PS22/9)** — limit must be in customer's best interest, not just technically affordable
- **POCA 2002** — structuring detection; SAR submission to NCA where suspicion exists
- **MLR 2017** — Enhanced Due Diligence mandatory for all PEP relationships
- **CIFAS / National Hunter** — UK shared fraud prevention databases
- **PRA SR 26-2** — model risk management for AI in credit decisions

---

*UK credit card acquisition platform — generic, no client name in codebase.*
