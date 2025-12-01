import io
import re
import json
import tempfile
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image

# Document parsers
import pdfplumber
import docx2txt
import pytesseract

# NLP
import spacy

# Optional semantic search / embeddings (comment out if not available)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_MODEL_AVAILABLE = True
except Exception:
    EMBEDDING_MODEL_AVAILABLE = False

nlp = spacy.load("en_core_web_sm")

# ---------- Config / policy sets ----------
POLICY_KEYWORDS = {
    "confidential": ["confidential", "internal use only", "do not distribute"],
    "financial": ["invoice", "bank account", "routing", "payment", "salary", "salary"],
    "personal": ["social security", "ssn", "passport", "aadhar", "pan", "phone", "email", "address"],
    "medical": ["diagnosis", "prescription", "medical", "doctor", "treatment"],
}

PII_REGEXES = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\+?\d[\d\-\s]{6,}\d",
    "ssn_like": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
    # Add country-specific patterns as needed
}

# Optional: load embedding model
EMBEDDING_MODEL = None
if EMBEDDING_MODEL_AVAILABLE:
    try:
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        EMBEDDING_MODEL = None

# ---------- Helpers: text extraction ----------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = None
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        text = docx2txt.process(tmp.name)
    return text or ""


def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    # Use pytesseract to OCR image
    text = pytesseract.image_to_string(image)
    return text

# ---------- PII detection and NER ----------

def detect_pii(text: str) -> Dict[str, List[Tuple[int,int,str]]]:
    """Return dict mapping PII type to list of tuples (start, end, match_text)"""
    findings = {k: [] for k in PII_REGEXES.keys()}
    for label, pattern in PII_REGEXES.items():
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            findings[label].append((m.start(), m.end(), m.group(0)))

    # Use spaCy NER to detect PERSON, ORG, GPE, etc.
    doc = nlp(text)
    spacy_findings = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "GPE", "LOC", "ORG", "DATE"):
            spacy_findings.append((ent.start_char, ent.end_char, ent.text, ent.label_))

    return {"regex": findings, "spacy": spacy_findings}

# ---------- Keyword / policy screening ----------

def policy_matches(text: str) -> Dict[str, List[str]]:
    matches = {k: [] for k in POLICY_KEYWORDS.keys()}
    lower = text.lower()
    for policy, keys in POLICY_KEYWORDS.items():
        for k in keys:
            if k.lower() in lower:
                matches[policy].append(k)
    return matches

# ---------- Risk scoring ----------

def compute_risk_score(matches: Dict[str, List[str]], pii: Dict) -> float:
    """Compute a simple risk score between 0 and 100."""
    score = 0.0
    # increase for number of keyword matches
    num_keyword_matches = sum(len(v) for v in matches.values())
    score += min(30, num_keyword_matches * 5)

    # increase for pii matches
    num_pii = sum(len(v) for v in pii['regex'].values())
    num_spacy = len(pii['spacy'])
    total_pii = num_pii + num_spacy
    score += min(50, total_pii * 10)

    # normalize to 0-100
    return min(100.0, score)

# ---------- Redaction ----------

def redact_text(text: str, pii: Dict, redact_with: str = "[REDACTED]") -> str:
    # Build list of spans to redact
    spans = []
    for _, items in pii['regex'].items():
        for s, e, _ in items:
            spans.append((s, e))
    for s, e, _, _ in pii['spacy']:
        spans.append((s, e))
    # merge and redact
    spans = sorted(spans, key=lambda x: x[0])
    out = []
    last = 0
    for s, e in spans:
        if s < last:
            continue
        out.append(text[last:s])
        out.append(redact_with)
        last = e
    out.append(text[last:])
    return ''.join(out)

# ---------- Optional semantic similarity search ----------

def semantic_find(text: str, queries: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    if EMBEDDING_MODEL is None:
        return []
    docs = [text]
    doc_emb = EMBEDDING_MODEL.encode(docs)
    q_embs = EMBEDDING_MODEL.encode(queries)
    sims = cosine_similarity(q_embs, doc_emb).flatten()
    return list(zip(queries, sims.tolist()))

# ---------- Report generation ----------

def generate_report(metadata: Dict, text: str, matches: Dict, pii: Dict, score: float) -> Dict:
    return {
        'metadata': metadata,
        'summary': {
            'risk_score': score,
            'keyword_matches': {k: len(v) for k, v in matches.items()},
            'pii_counts': {k: len(v) for k, v in pii['regex'].items()},
            'spacy_entities': len(pii['spacy'])
        },
        'matches': matches,
        'pii': pii,
        'text_snippet': text[:1000]
    }

# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="AI Document Screening", layout='wide')
    st.title("AI-based Document Screening Tool")
    st.write("Upload documents (PDF/DOCX/TXT/Images) and run a policy & PII screen.")

    uploaded = st.file_uploader("Upload file", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], accept_multiple_files=False)

    if not uploaded:
        st.info("Upload a file to get started.")
        return

    raw = uploaded.read()
    metadata = {
        'filename': uploaded.name,
        'type': uploaded.type,
        'size': len(raw)
    }

    # extract text based on file type
    text = ""
    if uploaded.type == 'application/pdf' or uploaded.name.lower().endswith('.pdf'):
        try:
            text = extract_text_from_pdf(raw)
        except Exception as e:
            st.error(f"PDF text extraction failed: {e}")
    elif uploaded.name.lower().endswith('.docx'):
        text = extract_text_from_docx(raw)
    elif uploaded.name.lower().endswith('.txt'):
        text = raw.decode('utf-8', errors='ignore')
    else:  # images
        try:
            text = extract_text_from_image(raw)
        except Exception as e:
            st.error(f"Image OCR failed: {e}")

    if not text.strip():
        st.warning("No text extracted from the file. Consider trying OCR (image) or checking the file.")

    st.subheader("Extracted Text (preview)")
    st.text_area("text_preview", value=text[:5000], height=240)

    # Screening
    with st.spinner("Running screening..."):
        matches = policy_matches(text)
        pii = detect_pii(text)
        score = compute_risk_score(matches, pii)

    st.metric("Risk score", f"{score:.1f}/100")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Policy keyword matches")
        for k, v in matches.items():
            st.write(f"**{k}** — {len(v)} matches: {v}")

        st.subheader("Detected PII (regex)")
        for k, items in pii['regex'].items():
            if items:
                st.write(f"{k}: {len(items)}")
                for _, _, text_match in items[:10]:
                    st.write(f" - {text_match}")

        st.subheader("Named Entities (spaCy)")
        for s, e, ent_text, ent_label in pii['spacy'][:50]:
            st.write(f"{ent_label}: {ent_text}")

    with col2:
        st.subheader("Actions")
        redact_button = st.button("Redact PII and download redacted text")
        if redact_button:
            redacted = redact_text(text, pii)
            st.download_button("Download redacted.txt", data=redacted, file_name=f"redacted_{uploaded.name}.txt")

        st.write("---")
        st.subheader("Export report")
        report = generate_report(metadata, text, matches, pii, score)
        st.download_button("Download JSON report", data=json.dumps(report, indent=2), file_name=f"report_{uploaded.name}.json")

    st.write("---")
    st.subheader("Notes & next steps")
    st.markdown(
        """
        - This tool is a starting point: adapt `POLICY_KEYWORDS` and `PII_REGEXES` for your organization's needs.
        - For higher accuracy, add a fine-tuned classifier (use the `sentence-transformers` embedding + sklearn or a supervised transformer model).
        - Add audit logging and user authentication before using in production.
        - Consider a secure environment for storing any uploaded documents (this demo does not persist files long-term).
        """
    )

if __name__ == '__main__':
    main()
