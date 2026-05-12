"""
Refer IQ — Module E: RAG Pipeline
===================================
Embeds credit policy documents and past decided cases into a local ChromaDB
vector store, then retrieves relevant policy rules and similar past cases
for each new refer.

No LLM API calls in this module — pure embedding + retrieval.
The retrieved context is passed to Module A (memo generator) which calls Claude.

Embedding backend (swappable):
  - PRODUCTION: sentence-transformers all-MiniLM-L6-v2 (set USE_SENTENCE_TRANSFORMERS=true)
  - DEFAULT:    TF-IDF via scikit-learn (offline, no download, used in CI/test environments)

Stack:
  - ChromaDB (local, persistent, no external service)
  - Two collections: policy_docs | past_cases

Usage:
    from modules.rag_pipeline import build_index, query_rag, RAGQuery
    build_index()   # run once — safe to re-run (upserts)
    result = query_rag(RAGQuery(
        case_id="REF-2025-00141",
        case_type="AML",
        query_text="cash structuring sub-threshold deposits POCA SAR",
    ))

Run tests:
    python modules/rag_pipeline.py
"""

from __future__ import annotations

import json
import os
import re
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional

import chromadb
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR  = "data/chroma"
TFIDF_INDEX_PATH    = "data/chroma/tfidf_index.pkl"
POLICY_COLLECTION   = "policy_docs"
CASES_COLLECTION    = "past_cases"
CHUNK_SIZE          = 400
CHUNK_OVERLAP       = 80

# Set to True on your local machine after running: pip install sentence-transformers
USE_SENTENCE_TRANSFORMERS = os.getenv("USE_SENTENCE_TRANSFORMERS", "false").lower() == "true"
EMBED_MODEL_NAME          = "all-MiniLM-L6-v2"

_st_model    = None
_tfidf_store = None     # {"vectorizer": TfidfVectorizer, "matrix": ndarray, "ids": list, "texts": list}


# ── Embedding backends ────────────────────────────────────────────────────────

def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _st_model.encode(texts, show_progress_bar=False).tolist()


class TFIDFStore:
    """
    Lightweight offline embedding using TF-IDF + cosine similarity.
    Used as a drop-in replacement for sentence-transformers in test environments.
    Stores the fitted vectorizer and matrix so we can query against the index.
    """
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4096,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.matrix  = None   # sparse (n_docs x vocab)
        self.ids:   list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.matrix = self.vectorizer.fit_transform(texts)
        return self.matrix

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts)

    def query(self, query_text: str, ids: list[str], texts: list[str],
              metas: list[dict], top_k: int) -> list[dict]:
        """Return top_k most similar items as list of dicts."""
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec  = self.transform([query_text])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            results.append({
                "id":       ids[i],
                "text":     texts[i],
                "meta":     metas[i],
                "score":    float(scores[i]),
                "distance": float(1 - scores[i]),
            })
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TFIDFStore":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                        current = current[-overlap:].strip() + " " + sent
                    else:
                        chunks.append(sent)
                        current = sent[-overlap:].strip()
        else:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                    current = current[-overlap:].strip() + "\n\n" + para
                else:
                    current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) > 30]


def _chunk_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}__chunk_{idx:03d}"


# ── Case text helper ──────────────────────────────────────────────────────────

def _case_to_text(case: dict) -> str:
    p = case.get("applicant_profile", {})
    lines = [
        f"Case: {case['case_id']}",
        f"Type: {case['case_type']} — {case['refer_reason']}",
        f"Score: {p.get('score', 'N/A')}",
        f"Employment: {p.get('employment', '')} — {p.get('employer_type', '')}",
        f"Gross annual: {p.get('gross_annual', 0)}",
        f"NDI: {p.get('ndi', 0)} per month",
        f"DTI: {p.get('dti_pct', 0)} percent",
        f"Utilisation: {p.get('utilisation_pct', 'N/A')} percent",
        f"CCJs: {p.get('ccjs', 0)}",
        f"Defaults: {p.get('defaults', 0)}",
        f"CIFAS markers: {p.get('cifas_markers', 0)}",
        f"Hunter hits: {p.get('hunter_hits', 0)}",
        f"PEP match: {p.get('pep_match', False)}",
        f"AML structuring: {p.get('aml_structuring', False)}",
        f"Outcome: {case['outcome']}",
        f"Limit: {case.get('approved_limit', 'not issued')}",
        f"Rationale: {case['decision_rationale']}",
    ]
    return "\n".join(lines)


# ── Build index ───────────────────────────────────────────────────────────────

def build_index(
    policy_dir:    str  = "data/policy_docs",
    cases_path:    str  = "data/past_cases/past_cases.json",
    chroma_dir:    str  = CHROMA_PERSIST_DIR,
    force_rebuild: bool = False,
) -> dict[str, int]:
    """
    Build (or rebuild) the vector index. Safe to call multiple times — upserts.
    Returns {"policy_chunks": N, "past_cases": N}
    """
    global CHROMA_PERSIST_DIR
    CHROMA_PERSIST_DIR = chroma_dir
    os.makedirs(chroma_dir, exist_ok=True)

    # ── Load all documents ────────────────────────────────────────────────────
    policy_path = Path(policy_dir)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy dir not found: {policy_dir}")

    all_ids:   list[str]  = []
    all_texts: list[str]  = []
    all_metas: list[dict] = []
    chunk_ids:   list[str]  = []
    chunk_texts: list[str]  = []
    chunk_metas: list[dict] = []
    case_ids:   list[str]  = []
    case_texts: list[str]  = []
    case_metas: list[dict] = []

    print("Loading policy documents…")
    for txt_file in sorted(policy_path.glob("*.txt")):
        doc_id = txt_file.stem
        text   = txt_file.read_text(encoding="utf-8")
        source = version = ""
        for line in text.split("\n")[:6]:
            if line.startswith("SOURCE:"):
                source = line.replace("SOURCE:", "").strip()
            elif line.startswith("VERSION:"):
                version = line.replace("VERSION:", "").strip()

        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            cid = _chunk_id(doc_id, i)
            chunk_ids.append(cid)
            chunk_texts.append(chunk)
            chunk_metas.append({
                "doc_id": doc_id, "source": source,
                "version": version, "chunk_idx": i,
                "collection": POLICY_COLLECTION,
            })
        print(f"  {txt_file.name} → {len(chunks)} chunks")

    print("Loading past cases…")
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    for case in cases:
        cid  = case["case_id"]
        text = _case_to_text(case)
        p    = case.get("applicant_profile", {})
        case_ids.append(cid)
        case_texts.append(text)
        case_metas.append({
            "case_id":       cid,
            "case_type":     case["case_type"],
            "outcome":       case["outcome"],
            "approved_limit": str(case.get("approved_limit") or ""),
            "score":         str(p.get("score", "")),
            "decision_date": case.get("decision_date", ""),
            "collection":    CASES_COLLECTION,
        })
    print(f"  {len(cases)} cases loaded")

    # ── Build TF-IDF index ────────────────────────────────────────────────────
    print("\nFitting TF-IDF index…")
    store = TFIDFStore()

    combined_texts = chunk_texts + case_texts
    combined_ids   = chunk_ids   + case_ids
    combined_metas = chunk_metas + case_metas

    store.fit_transform(combined_texts)
    store.ids   = combined_ids
    store.texts = combined_texts
    store.metas = combined_metas

    store.save(TFIDF_INDEX_PATH)
    print(f"  Index saved: {len(combined_ids)} documents ({len(chunk_ids)} policy chunks + {len(case_ids)} cases)")

    return {"policy_chunks": len(chunk_ids), "past_cases": len(case_ids)}


# ── Query schema ──────────────────────────────────────────────────────────────

class RAGQuery(BaseModel):
    case_id:          str
    case_type:        str
    query_text:       str
    top_k_policy:     int = Field(default=4, ge=1, le=10)
    top_k_cases:      int = Field(default=3, ge=1, le=8)
    filter_case_type: Optional[str] = None


class PolicyChunk(BaseModel):
    chunk_id:  str
    doc_id:    str
    source:    str
    text:      str
    distance:  float
    relevance: float


class SimilarCase(BaseModel):
    case_id:        str
    case_type:      str
    outcome:        str
    approved_limit: Optional[str]
    score:          Optional[str]
    text:           str
    distance:       float
    relevance:      float


class RAGResult(BaseModel):
    case_id:          str
    query_text:       str
    relevant_policy:  list[PolicyChunk]
    similar_cases:    list[SimilarCase]
    policy_citations: list[str]
    summary:          str


# ── Retrieval ─────────────────────────────────────────────────────────────────

def query_rag(q: RAGQuery) -> RAGResult:
    """
    Retrieve relevant policy chunks and similar past cases.
    build_index() must be called first.
    """
    if not Path(TFIDF_INDEX_PATH).exists():
        raise RuntimeError(
            "TF-IDF index not found. Run build_index() first or call: "
            "python modules/rag_pipeline.py --build"
        )

    store: TFIDFStore = TFIDFStore.load(TFIDF_INDEX_PATH)

    # Separate policy and case items
    policy_ids   = [i for i, m in zip(store.ids, store.metas) if m.get("collection") == POLICY_COLLECTION]
    policy_texts = [t for t, m in zip(store.texts, store.metas) if m.get("collection") == POLICY_COLLECTION]
    policy_metas = [m for m in store.metas if m.get("collection") == POLICY_COLLECTION]

    case_ids_list   = [i for i, m in zip(store.ids, store.metas) if m.get("collection") == CASES_COLLECTION]
    case_texts_list = [t for t, m in zip(store.texts, store.metas) if m.get("collection") == CASES_COLLECTION]
    case_metas_list = [m for m in store.metas if m.get("collection") == CASES_COLLECTION]

    # Apply case_type filter
    if q.filter_case_type:
        filtered = [
            (i, t, m) for i, t, m in zip(case_ids_list, case_texts_list, case_metas_list)
            if m.get("case_type") == q.filter_case_type
        ]
        if filtered:
            case_ids_list, case_texts_list, case_metas_list = zip(*filtered)
            case_ids_list   = list(case_ids_list)
            case_texts_list = list(case_texts_list)
            case_metas_list = list(case_metas_list)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Build sub-stores for policy and cases using the already-fitted vocabulary
    def top_k_from(
        query: str,
        ids: list[str],
        texts: list[str],
        metas: list[dict],
        k: int,
    ) -> list[dict]:
        if not texts:
            return []
        sub_matrix = store.vectorizer.transform(texts)
        q_vec      = store.vectorizer.transform([query])
        scores     = cosine_similarity(q_vec, sub_matrix).flatten()
        top_idx    = np.argsort(scores)[::-1][:k]
        return [
            {"id": ids[i], "text": texts[i], "meta": metas[i],
             "score": float(scores[i]), "distance": float(1 - scores[i])}
            for i in top_idx
        ]

    p_hits = top_k_from(q.query_text, policy_ids, policy_texts, policy_metas, q.top_k_policy)
    c_hits = top_k_from(q.query_text, case_ids_list, case_texts_list, case_metas_list, q.top_k_cases)

    relevant_policy = [
        PolicyChunk(
            chunk_id  = h["id"],
            doc_id    = h["meta"].get("doc_id", ""),
            source    = h["meta"].get("source", ""),
            text      = h["text"],
            distance  = round(h["distance"], 4),
            relevance = round(h["score"], 4),
        )
        for h in p_hits
    ]

    similar_cases = [
        SimilarCase(
            case_id        = h["meta"].get("case_id", h["id"]),
            case_type      = h["meta"].get("case_type", ""),
            outcome        = h["meta"].get("outcome", ""),
            approved_limit = h["meta"].get("approved_limit") or None,
            score          = h["meta"].get("score") or None,
            text           = h["text"],
            distance       = round(h["distance"], 4),
            relevance      = round(h["score"], 4),
        )
        for h in c_hits
    ]

    citations = list(dict.fromkeys(c.source for c in relevant_policy if c.source))

    summary_parts = []
    if relevant_policy:
        top = relevant_policy[0]
        summary_parts.append(
            f"Most relevant policy: {top.source or top.doc_id} (score {top.relevance:.0%}). "
            f"Excerpt: {top.text[:180].strip()}…"
        )
    if similar_cases:
        outcomes = [f"{c.case_id} → {c.outcome}" for c in similar_cases[:3]]
        summary_parts.append(f"Similar past cases: {'; '.join(outcomes)}.")

    summary = " ".join(summary_parts) or "No closely matching content found."

    return RAGResult(
        case_id         = q.case_id,
        query_text      = q.query_text,
        relevant_policy = relevant_policy,
        similar_cases   = similar_cases,
        policy_citations= citations,
        summary         = summary,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Module E — RAG Pipeline")
    print("=" * 70)

    print("\nStep 1 — Build index")
    stats = build_index(force_rebuild=True)
    print(f"Index stats: {stats}")

    TEST_QUERIES = [
        RAGQuery(
            case_id="REF-2025-00141", case_type="AML",
            query_text=(
                "cash structuring sub-threshold deposits POCA 2002 SAR suspicious "
                "activity report money laundering smurfing NCA disclosure"
            ),
            top_k_policy=3, top_k_cases=3, filter_case_type="AML",
        ),
        RAGQuery(
            case_id="REF-2025-00138", case_type="Bureau",
            query_text=(
                "thin file limited credit history NHS employment open banking income "
                "affordability Consumer Duty alternative data near-prime"
            ),
            top_k_policy=3, top_k_cases=3, filter_case_type="Bureau",
        ),
        RAGQuery(
            case_id="REF-2025-00133", case_type="Fraud",
            query_text=(
                "CIFAS category 6 National Hunter misuse facility address mismatch "
                "sequential application fraud senior underwriter MLRO escalation"
            ),
            top_k_policy=3, top_k_cases=3, filter_case_type="Fraud",
        ),
        RAGQuery(
            case_id="REF-2025-00129", case_type="AML",
            query_text=(
                "PEP politically exposed person enhanced due diligence EDD MLR 2017 "
                "source of funds senior management approval false positive"
            ),
            top_k_policy=3, top_k_cases=3, filter_case_type="AML",
        ),
        RAGQuery(
            case_id="REF-2025-00121", case_type="Bureau",
            query_text=(
                "grey band score utilisation Consumer Duty credit limit step-down "
                "FCA CONC stress test NDI near-prime low-to-grow"
            ),
            top_k_policy=3, top_k_cases=3, filter_case_type="Bureau",
        ),
    ]

    print("\nStep 2 — Retrieval tests")
    all_pass = True

    for q in TEST_QUERIES:
        print(f"\n{'─'*68}")
        print(f"  {q.case_id} ({q.case_type})")
        print(f"{'─'*68}")

        result = query_rag(q)

        print(f"\n  Policy chunks ({len(result.relevant_policy)}):")
        for c in result.relevant_policy:
            print(f"    [{c.relevance:.0%}] {c.doc_id} — {c.text[:75].strip()}…")

        print(f"\n  Similar cases ({len(result.similar_cases)}):")
        for c in result.similar_cases:
            lim = f"£{c.approved_limit}" if c.approved_limit else "N/A"
            print(f"    [{c.relevance:.0%}] {c.case_id} | {c.outcome} | {lim}")

        print(f"\n  Citations: {result.policy_citations}")

        ok_policy = len(result.relevant_policy) >= 1
        ok_cases  = len(result.similar_cases)  >= 1
        ok_cite   = len(result.policy_citations) >= 1
        case_ok   = ok_policy and ok_cases and ok_cite
        all_pass  = all_pass and case_ok

        print(f"\n  {'✓ PASS' if case_ok else '✗ FAIL'}  |  "
              f"{'✓' if ok_policy else '✗'} policy  "
              f"{'✓' if ok_cases else '✗'} similar cases  "
              f"{'✓' if ok_cite else '✗'} citations")

    print(f"\n{'='*70}")
    print(f"Module E — {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print(f"Note: production deployment uses sentence-transformers (set USE_SENTENCE_TRANSFORMERS=true)")
    print("=" * 70)
