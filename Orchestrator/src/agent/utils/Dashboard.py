import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta
import io

import nest_asyncio
import streamlit as st
from pymongo import MongoClient, DESCENDING

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedFactCheck",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background:#060b18;}
[data-testid="stAppViewContainer"]{background:#060b18;}
[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:#080d1c!important;border-right:1px solid rgba(99,179,237,0.12)!important;}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span{color:#94a3b8!important;}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{color:#e2e8f0!important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#0f172a;}
::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:3px;}
.stTextInput input,.stTextArea textarea{background:#0f1c2e!important;border:1px solid rgba(99,179,237,0.2)!important;color:#e2e8f0!important;border-radius:8px!important;}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:rgba(99,179,237,0.6)!important;box-shadow:0 0 0 3px rgba(99,179,237,0.1)!important;}
.stButton>button{background:linear-gradient(135deg,#1a3a5c,#0f2a46)!important;color:#7dd3fc!important;border:1px solid rgba(99,179,237,0.25)!important;border-radius:8px!important;font-size:13px!important;font-weight:500!important;transition:all 0.2s!important;}
.stButton>button:hover{background:linear-gradient(135deg,#1e4d7a,#163660)!important;border-color:rgba(99,179,237,0.5)!important;box-shadow:0 0 16px rgba(99,179,237,0.15)!important;transform:translateY(-1px)!important;}
button[kind="primary"]{background:linear-gradient(135deg,#0ea5e9,#0284c7)!important;color:#fff!important;border:none!important;box-shadow:0 4px 20px rgba(14,165,233,0.3)!important;}
div[role="radiogroup"]{gap:6px!important;}
.stRadio label{background:#0f1c2e!important;border:1px solid rgba(99,179,237,0.15)!important;border-radius:8px!important;padding:6px 14px!important;color:#64748b!important;font-size:13px!important;transition:all 0.15s!important;}
.stRadio label:hover{border-color:rgba(99,179,237,0.4)!important;color:#94a3b8!important;}
hr{border-color:rgba(99,179,237,0.1)!important;margin:12px 0!important;}
.stAlert{border-radius:10px!important;}
.page-title{font-size:11px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#334155;margin-bottom:1.2rem;}
.sidebar-logo{display:flex;align-items:center;gap:10px;margin-bottom:20px;}
.sidebar-icon{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#0ea5e9,#0284c7);display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 16px rgba(14,165,233,0.35);}
.sidebar-brand{font-size:15px;font-weight:700;color:#e2e8f0!important;}
.sidebar-tagline{font-size:10px;color:#334155!important;letter-spacing:0.04em;}
.metrics-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
.metric-card{background:linear-gradient(145deg,#0c1829,#0a1520);border:1px solid rgba(99,179,237,0.12);border-radius:12px;padding:12px 14px;position:relative;overflow:hidden;}
.metric-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;border-radius:12px 12px 0 0;}
.mc-total::before{background:linear-gradient(90deg,#334155,#475569);}
.mc-sup::before{background:linear-gradient(90deg,#14532d,#22c55e);}
.mc-ref::before{background:linear-gradient(90deg,#7f1d1d,#ef4444);}
.mc-nei::before{background:linear-gradient(90deg,#7c2d12,#f97316);}
.metric-label{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;}
.metric-val{font-size:26px;font-weight:700;line-height:1;}
.mv-total{color:#e2e8f0;}.mv-sup{color:#4ade80;}.mv-ref{color:#f87171;}.mv-nei{color:#fb923c;}
.glow-sep{height:1px;background:linear-gradient(90deg,transparent,rgba(14,165,233,0.3),transparent);margin:14px 0;}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:600;padding:3px 8px;border-radius:20px;letter-spacing:0.03em;}
.b-sup{background:rgba(34,197,94,0.12);color:#4ade80;border:1px solid rgba(34,197,94,0.25);}
.b-ref{background:rgba(239,68,68,0.12);color:#f87171;border:1px solid rgba(239,68,68,0.25);}
.b-nei{background:rgba(249,115,22,0.12);color:#fb923c;border:1px solid rgba(249,115,22,0.25);}
.b-src{background:rgba(99,179,237,0.08);color:#7dd3fc;border:1px solid rgba(99,179,237,0.2);}
.claim-item{background:#0c1829;border:1px solid rgba(99,179,237,0.1);border-radius:10px;padding:11px 13px;margin-bottom:7px;transition:all 0.18s;}
.claim-item:hover{background:#0f2040;border-color:rgba(99,179,237,0.3);box-shadow:0 0 12px rgba(14,165,233,0.07);}
.claim-item.active{background:#0f2a4a;border-color:#0ea5e9;box-shadow:0 0 20px rgba(14,165,233,0.12);}
.ci-text{font-size:12px;color:#94a3b8;line-height:1.45;margin-bottom:7px;}
.claim-item.active .ci-text{color:#bae6fd;}
.ci-foot{display:flex;align-items:center;justify-content:space-between;}
.ci-ts{font-size:10px;color:#334155;font-family:'JetBrains Mono',monospace;}
.result-header{background:linear-gradient(145deg,#0a1625,#0c1e35);border:1px solid rgba(99,179,237,0.12);border-radius:14px;padding:20px 22px;margin-bottom:16px;position:relative;overflow:hidden;}
.result-header::before{content:"VERIFIED CLAIM";position:absolute;right:16px;top:14px;font-size:9px;letter-spacing:0.15em;color:#1e3a5f;font-weight:700;}
.rh-text{font-size:15px;color:#cbd5e1;line-height:1.6;margin-bottom:14px;}
.conf-track{height:4px;border-radius:2px;background:rgba(255,255,255,0.06);overflow:hidden;margin-bottom:3px;}
.conf-fill{height:100%;border-radius:2px;}
.cf-sup{background:linear-gradient(90deg,#14532d,#22c55e);}
.cf-ref{background:linear-gradient(90deg,#7f1d1d,#ef4444);}
.cf-nei{background:linear-gradient(90deg,#7c2d12,#f97316);}
.conf-pct{font-size:11px;color:#475569;font-family:'JetBrains Mono',monospace;}
.verdict-card{border-radius:14px;padding:18px 20px;margin-bottom:12px;}
.vc-sup{background:linear-gradient(145deg,#071c10,#052e16);border:1px solid rgba(34,197,94,0.25);}
.vc-ref{background:linear-gradient(145deg,#1c0707,#450a0a);border:1px solid rgba(239,68,68,0.25);}
.vc-nei{background:linear-gradient(145deg,#1c0f05,#431407);border:1px solid rgba(249,115,22,0.25);}
.vc-label{font-size:10px;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;margin-bottom:8px;}
.vc-sup .vc-label{color:#4ade80;}.vc-ref .vc-label{color:#f87171;}.vc-nei .vc-label{color:#fb923c;}
.vc-text{font-size:14px;color:#e2e8f0;line-height:1.6;margin-bottom:12px;}
.ev-card{background:#0a1625;border:1px solid rgba(99,179,237,0.1);border-radius:12px;padding:14px 16px;margin-bottom:10px;transition:border-color 0.18s;}
.ev-card:hover{border-color:rgba(99,179,237,0.25);}
.ev-title{font-size:13px;font-weight:600;color:#cbd5e1;margin-bottom:6px;line-height:1.4;}
.ev-meta{font-size:11px;color:#475569;display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;}
.ev-text{font-size:12px;color:#64748b;line-height:1.65;}
.dir-dot{width:7px;height:7px;border-radius:50%;display:inline-block;flex-shrink:0;margin-top:3px;}
.dd-sup{background:#22c55e;box-shadow:0 0 6px #22c55e88;}
.dd-ref{background:#ef4444;box-shadow:0 0 6px #ef444488;}
.dd-neu{background:#64748b;}
.pmid-lnk{color:#38bdf8;font-size:11px;font-family:'JetBrains Mono',monospace;}
.cot-box{background:#080e1a;border:1px solid rgba(99,179,237,0.12);border-left:3px solid #0ea5e9;border-radius:0 10px 10px 0;padding:14px 16px;font-size:13px;color:#94a3b8;line-height:1.75;font-family:'JetBrains Mono',monospace;margin-bottom:12px;white-space:pre-wrap;}
.sub-lbl{font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#475569;margin-bottom:6px;}
.sub-text{font-size:13px;color:#64748b;margin-bottom:8px;line-height:1.5;}
.trace-wrap{position:relative;padding-left:24px;}
.trace-wrap::before{content:"";position:absolute;left:7px;top:10px;bottom:10px;width:1px;background:linear-gradient(180deg,#0ea5e9,transparent);}
.trace-step{position:relative;margin-bottom:18px;}
.trace-dot{position:absolute;left:-22px;top:3px;width:10px;height:10px;border-radius:50%;background:#0ea5e9;box-shadow:0 0 8px rgba(14,165,233,0.6);border:2px solid #0284c7;}
.trace-agent{font-size:13px;font-weight:600;color:#7dd3fc;margin-bottom:2px;}
.trace-model{font-size:12px;color:#94a3b8;line-height:1.5;margin-bottom:4px;white-space:pre-wrap;}
.trace-chip{font-size:10px;padding:2px 7px;border-radius:10px;background:rgba(99,179,237,0.08);color:#7dd3fc;border:1px solid rgba(99,179,237,0.15);margin-right:4px;}
.nv-header{background:linear-gradient(145deg,#0a1625,#0c1e35);border:1px solid rgba(99,179,237,0.15);border-radius:14px;padding:20px 22px;margin-bottom:16px;}
.nv-title{font-size:18px;font-weight:700;color:#e2e8f0;margin-bottom:4px;}
.nv-sub{font-size:13px;color:#475569;}
.pipeline-box{background:#050a14;border:1px solid rgba(99,179,237,0.1);border-radius:10px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#475569;line-height:1.9;margin-top:12px;}
.pl-ok{color:#4ade80;}.pl-run{color:#38bdf8;}.pl-err{color:#f87171;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE API SERVER
# ─────────────────────────────────────────────────────────────────────────────
# La dashboard ora funge solo da client leggero
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

# ─────────────────────────────────────────────────────────────────────────────
# CONNESSIONE MONGODB
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    uri     = st.secrets.get("MONGO_URI",     os.getenv("MONGO_URI",     "mongodb://localhost:27017/"))
    db_name = st.secrets.get("MONGO_DB_NAME", os.getenv("MONGO_DB_NAME", "medfactcheck"))
    return MongoClient(uri, serverSelectionTimeoutMS=4000)[db_name]

def col(name):
    return get_db()[name]

@st.cache_data(ttl=3600, max_entries=50)
def get_results(verdict_filter=None, kw=None, date_filter=None, source_filter=None, limit=50):
    """Interroga MongoDB per recuperare i claim, applicando filtri e ricerca su tutto il database."""
    query = {}
    if verdict_filter:
        query["final_verdict"] = verdict_filter
    if kw:
        query["original_text"] = {"$regex": kw, "$options": "i"}
    if date_filter and date_filter != "All time":
        now = datetime.now()
        if date_filter == "Last 24h":
            dt = now - timedelta(days=1)
        elif date_filter == "Last 7 days":
            dt = now - timedelta(days=7)
        elif date_filter == "Last 30 days":
            dt = now - timedelta(days=30)
        query["timestamp"] = {"$gte": dt.isoformat()}
        
    if source_filter and source_filter != "All":
        ev_docs = list(get_db()["evidence"].find({"source": source_filter}, {"claim_id": 1}))
        claim_ids = [d["claim_id"] for d in ev_docs]
        query["claim_id"] = {"$in": claim_ids}

    return list(col("final_results").find(query, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit))

@st.cache_data(ttl=3600, max_entries=10)
def get_evidence(claim_id):
    return list(col("evidence").find({"claim_id": claim_id}, {"_id": 0}).sort("rrf_score", DESCENDING))

@st.cache_data(ttl=3600)
def get_stats():
    total = col("final_results").count_documents({})
    if total == 0:
        return {"total": 0, "sup": 0, "ref": 0, "nei": 0, "sources": []}
    
    # Raggruppamento robusto case-insensitive per evitare disallineamenti di etichette
    agg_raw = col("final_results").aggregate([
        {"$group": {"_id": "$final_verdict", "count": {"$sum": 1}}}
    ])
    
    counts = {"sup": 0, "ref": 0, "nei": 0}
    for r in agg_raw:
        label = str(r["_id"]).upper()
        if label == "SUPPORTED": counts["sup"] += r["count"]
        elif label == "REFUTED": counts["ref"] += r["count"]
        else: counts["nei"] += r["count"]

    return {
        "total"  : total,
        "sup"    : counts["sup"],
        "ref"    : counts["ref"],
        "nei"    : counts["nei"],
        "sources": col("evidence").distinct("source"),
    }

def reload_claim(claim_id):
    """Rilegge il documento aggiornato da MongoDB dopo l'esecuzione della pipeline."""
    return col("final_results").find_one({"claim_id": claim_id}, {"_id": 0})

def generate_pdf_report(claim_data):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return b"Error: reportlab is not installed. Please run 'pip install reportlab' to enable PDF generation."

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    story = []
    story.append(Paragraph("MedFactCheck - Fact-Checking Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Original Claim:</b> {claim_data.get('original_text', '')}", normal_style))
    story.append(Spacer(1, 12))
    
    verdict = claim_data.get('final_verdict', 'Unknown')
    conf = int(claim_data.get('avg_confidence', 0) * 100)
    story.append(Paragraph(f"<b>Final Verdict:</b> {verdict} ({conf}% confidence)", normal_style))
    story.append(Spacer(1, 24))
    
    story.append(Paragraph("<b>Analysis Breakdown:</b>", h2_style))
    for i, sv in enumerate(claim_data.get('sub_verdicts', []), 1):
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<u>Sub-claim {i}:</u> {sv.get('sub_claim', '')}", normal_style))
        story.append(Paragraph(f"<b>Verdict:</b> {sv.get('verdict', '')} ({int(sv.get('confidence_score', 0)*100)}%)", normal_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Reasoning:</b> {sv.get('chain_of_thought_log', '').replace('<', '&lt;')}", normal_style))
    
    story.append(Spacer(1, 24))
    story.append(Paragraph("<b>Supporting Evidence:</b>", h2_style))
    
    passages = get_evidence(claim_data.get("claim_id", ""))
    for i, p in enumerate(passages, 1):
        story.append(Spacer(1, 6))
        txt = (p.get("testo") or p.get("text") or p.get("abstract", ""))[:800] + "..."
        story.append(Paragraph(f"<b>[{i}] {p.get('title', '—')}</b> (Source: {p.get('source', '')})", normal_style))
        story.append(Paragraph(f"<i>{txt}</i>", normal_style))
        
    doc.build(story)
    return buffer.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(claim_text: str, image_path: str = None) -> dict:
    """
    Invia il claim all'API FastAPI in esecuzione sul server.
    """
    import requests

    data = {"text": claim_text}
    files = {}
    if image_path:
        files["image"] = open(image_path, "rb")
        
    try:
        response = requests.post(f"{API_URL}/verify", data=data, files=files, timeout=3600)
        response.raise_for_status()
        # Restituisce l'ID del claim, con cui recupereremo il risultato da Mongo
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API connection error. Make sure the server is running: {e}")
    finally:
        if "image" in files:
            files["image"].close()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
VC    = {"Supported": "sup", "Refuted": "ref", "Not Enough Information": "nei"}
ICON  = {"sup": "✦", "ref": "✕", "nei": "?"}
SHORT = {"Supported": "Supported", "Refuted": "Refuted", "Not Enough Information": "NEI"}

def badge(verdict):
    c = VC.get(verdict, "nei")
    return f'<span class="badge b-{c}">{ICON[c]} {SHORT.get(verdict, verdict)}</span>'

def conf_bar(score, vc):
    pct = int(score * 100)
    return (
        f'<div class="conf-track">'
        f'<div class="conf-fill cf-{vc}" style="width:{pct}%"></div></div>'
        f'<span class="conf-pct">{pct}%</span>'
    )

def fmt_ts(iso):
    try:    return iso[:16].replace("T", " ")
    except: return ""

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "sel": None, "sec": "Verdict",
    "flt": "All", "kw": "", "thr": 0,
    "flt_date": "All time", "flt_src": "All",
    "running": False, "error": None,
    "trigger_verify": False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-icon">🛡️</div>
      <div>
        <div class="sidebar-brand">MedFactCheck</div>
        <div class="sidebar-tagline">BIOMEDICAL FACT VERIFICATION</div>
      </div>
    </div>""", unsafe_allow_html=True)

    try:
        s = get_stats()
        st.markdown(f"""
        <div class="metrics-grid">
          <div class="metric-card mc-total">
            <div class="metric-label">Total</div>
            <div class="metric-val mv-total">{s['total']}</div>
          </div>
          <div class="metric-card mc-sup">
            <div class="metric-label">Supported</div>
            <div class="metric-val mv-sup">{s['sup']}</div>
          </div>
          <div class="metric-card mc-ref">
            <div class="metric-label">Refuted</div>
            <div class="metric-val mv-ref">{s['ref']}</div>
          </div>
          <div class="metric-card mc-nei">
            <div class="metric-label">NEI</div>
            <div class="metric-val mv-nei">{s['nei']}</div>
          </div>
        </div>
        <div style="margin-bottom:10px;">
          {''.join(f'<span class="badge b-src">{src}</span> ' for src in s['sources'])}
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"MongoDB: {e}")

    st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;color:#475569;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Filters</p>',
                unsafe_allow_html=True)

    flt = st.selectbox(
        "Verdict", ["All", "Supported", "Refuted", "Not Enough Information"],
        index=["All","Supported","Refuted","Not Enough Information"].index(st.session_state.flt),
        label_visibility="collapsed"
    )
    st.session_state.flt = flt

    sources = ["All"] + list(s['sources'])
    flt_src = st.selectbox(
        "Source", sources,
        index=sources.index(st.session_state.flt_src) if st.session_state.flt_src in sources else 0
    )
    st.session_state.flt_src = flt_src
    
    flt_date = st.selectbox(
        "Date", ["All time", "Last 24h", "Last 7 days", "Last 30 days"],
        index=["All time", "Last 24h", "Last 7 days", "Last 30 days"].index(st.session_state.flt_date)
    )
    st.session_state.flt_date = flt_date

    kw = st.text_input("🔍 Keyword", value=st.session_state.kw,
                       placeholder="e.g., vitamin D, mRNA…", label_visibility="collapsed")
    st.session_state.kw = kw

    thr = st.slider("Confidence threshold", 0, 100, int(st.session_state.thr), 5, format="%d%%")
    st.session_state.thr = thr

    st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BODY
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="medium")

# ═══════════════════════════
# SINISTRA — Ultime ricerche
# ═══════════════════════════
with left:
    st.markdown('<div class="page-title">🕐 Latest searches</div>', unsafe_allow_html=True)
    try:
        vf  = None if st.session_state.flt == "All" else st.session_state.flt
        sf  = None if st.session_state.flt_src == "All" else st.session_state.flt_src
        df  = None if st.session_state.flt_date == "All time" else st.session_state.flt_date
        results = get_results(vf, kw.strip() or None, df, sf, 50)
        results = [r for r in results if r.get("avg_confidence", 0) * 100 >= st.session_state.thr]

        if not results:
            st.markdown('<p style="color:#334155;font-size:13px;margin-top:8px;">No claims found.</p>',
                        unsafe_allow_html=True)
        else:
            for r in results:
                vc  = VC.get(r["final_verdict"], "nei")
                act = ("active" if st.session_state.sel
                       and st.session_state.sel.get("claim_id") == r.get("claim_id") else "")
                txt = r["original_text"][:85] + ("…" if len(r["original_text"]) > 85 else "")
                pct = int(r.get("avg_confidence", 0) * 100)

                st.markdown(
                    f'<div class="claim-item {act}">'
                    f'<div class="ci-text">{txt}</div>'
                    f'<div class="ci-foot">{badge(r["final_verdict"])}'
                    f'<span class="ci-ts">{pct}% · {fmt_ts(r.get("timestamp",""))}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True)

                if st.button("Open →", key=f"b_{r['claim_id']}", use_container_width=True):
                    st.session_state.sel     = r
                    st.session_state.sec     = "Verdict"
                    st.session_state.running = False
                    st.session_state.error   = None
                    st.rerun()

    except Exception as e:
        st.error(str(e))

# ══════════════════════════════════
# DESTRA — Input  /  Risultato
# ══════════════════════════════════
with right:

    # Top Bar con Bottone "New Check" in alto a destra
    top_c1, top_c2 = st.columns([4, 1])
    with top_c2:
        if st.button("＋ New check", use_container_width=True, key="btn_new_check_top"):
            st.session_state.sel     = None
            st.session_state.error   = None
            st.session_state.running = False
            st.session_state.trigger_verify = False
            st.rerun()

    # ── NESSUN CLAIM SELEZIONATO → input form ──────────────────────────────
    if st.session_state.sel is None:
        with top_c1:
            st.markdown('<div class="page-title">🔍 New check</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="nv-header">
          <div class="nv-title">Enter a biomedical claim</div>
          <div class="nv-sub">
            The LangGraph pipeline will decompose the claim, retrieve evidence from
            PubMed, PMC, and Knowledge Bases, generate a chain-of-thought reasoning and
            assign a verdict via DeBERTa-v3.
          </div>
        </div>""", unsafe_allow_html=True)

        claim_text = st.text_area(
            "", height=120,
            placeholder="Es: Vitamin D supplementation reduces the risk of COVID-19 infection.",
            label_visibility="collapsed",
            disabled=st.session_state.running,
            key="claim_text_input"
        )
        
        # Aggiunta uploader per immagini
        uploaded_image = st.file_uploader(
            "🖼️ Attach image (Optional for multimodal analysis)", 
            type=["png", "jpg", "jpeg"],
            disabled=st.session_state.running,
            key="claim_image_input"
        )

        can_submit = bool(claim_text.strip() or uploaded_image is not None)

        def handle_verify():
            st.session_state.running = True
            st.session_state.trigger_verify = True
            st.session_state.error = None

        c1, c2, _ = st.columns([2, 1, 3])
        with c1:
            run_btn = st.button(
                "▶  Verify claim", type="primary",
                use_container_width=True,
                disabled=st.session_state.running or not can_submit,
                on_click=handle_verify
            )
        with c2:
            if st.button("✕", use_container_width=True, disabled=st.session_state.running):
                st.session_state.error = None
                st.rerun()

        # ── Errore ────────────────────────────────────────────────────────
        if st.session_state.error:
            st.error(f"❌ Pipeline error: {st.session_state.error}")

        # ── Esecuzione ────────────────────────────────────────────────────
        if st.session_state.trigger_verify:
            st.session_state.trigger_verify = False
            
            st.markdown("""
            <div class="pipeline-box">
              <span class="pl-run">▶ Starting LangGraph pipeline…</span><br>
              <span class="pl-run">⚙ init_qwen · init_deberta · init_retrieval · init_db  [parallel]</span><br>
              <span style="color:#334155;">&nbsp;&nbsp;First execution: 1-3 min to load models.</span><br>
              <span class="pl-run">✦ save_claim → input_to_json → decompose</span><br>
              <span class="pl-run">✦ retrieve → save_evidence → reason</span><br>
              <span class="pl-run">✦ veracity → save_verdicts → print_final_result</span>
            </div>""", unsafe_allow_html=True)

            with st.spinner("Pipeline running…"):
                image_path = None
                try:
                    # Gestione sicura del file temporaneo
                    if uploaded_image:
                        file_ext = uploaded_image.name.split('.')[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                            tmp_file.write(uploaded_image.getvalue())
                            image_path = tmp_file.name

                    final = run_pipeline(claim_text.strip(), image_path)
                    
                    # Invalida la cache per forzare la lettura dei nuovi risultati aggiornati da MongoDB
                    st.cache_data.clear()
                    st.session_state.running = False

                    if final and final.get("claim_id"):
                        fresh = reload_claim(final["claim_id"])
                        st.session_state.sel = fresh or final
                        st.session_state.sec = "Verdict"
                    st.rerun()

                except Exception as e:
                    st.session_state.running = False
                    st.session_state.error   = str(e)
                    st.rerun()
                    
                finally:
                    # Pulizia GARANTITA del file temporaneo in memoria
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except OSError:
                            pass

        # ── Esempi rapidi ─────────────────────────────────────────────────
        if not st.session_state.running:
            st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:10px;font-weight:600;letter-spacing:0.1em;'
                        'text-transform:uppercase;color:#334155;margin-bottom:8px;">Quick examples</p>',
                        unsafe_allow_html=True)
            for ex in [
                "Vitamin D supplementation reduces the risk of COVID-19.",
                "mRNA vaccines alter human DNA and cause myocarditis.",
                "Aspirin reduces colorectal cancer risk.",
                "High sodium intake increases blood pressure.",
                "Statins reduce cardiovascular mortality in the elderly.",
            ]:
                st.markdown(f'<p style="font-size:12px;color:#475569;margin:4px 0;">↗ {ex}</p>',
                            unsafe_allow_html=True)

    # ── CLAIM SELEZIONATO → visualizzazione ────────────────────────────────
    else:
        with top_c1:
            st.markdown('<div class="page-title">📋 Verified result</div>', unsafe_allow_html=True)

        r  = st.session_state.sel
        verdetto = r.get("final_verdict")
        
        if not verdetto:
            st.error("⚠️ The server interrupted the analysis (Likely LLM token limit reached). No verdict generated.")
            if st.button("← Back"):
                st.session_state.sel = None
                st.rerun()
            st.stop()

        vc = VC.get(verdetto, "nei")

        st.markdown(
            f'<div class="result-header">'
            f'<div class="rh-text">{r["original_text"]}</div>'
            f'<div style="display:flex;align-items:center;gap:16px;">'
            f'{badge(verdetto)}'
            f'<div style="flex:1;max-width:200px;">{conf_bar(r.get("avg_confidence",0), vc)}</div>'
            f'</div></div>',
            unsafe_allow_html=True)

        back, pdf_col, *_ = st.columns([1, 2, 2])
        with back:
            if st.button("← Back"):
                st.session_state.sel   = None
                st.session_state.error = None
                st.rerun()
        with pdf_col:
            pdf_bytes = generate_pdf_report(r)
            st.download_button(
                label="📄 Export PDF Report",
                data=pdf_bytes,
                file_name=f"MedFactCheck_Report_{r.get('claim_id', 'unknown')}.pdf",
                mime="application/pdf"
            )

        st.markdown('<div class="glow-sep"></div>', unsafe_allow_html=True)

        # ── SELEZIONE SEZIONE ─────────────────────────────────────────────
        st.radio(
            "", ["Verdict", "Evidence", "Reasoning", "Agent trace"],
            horizontal=True,
            key="sec",
            label_visibility="collapsed",
        )
        st.markdown("")

        # ── VERDICT ──────────────────────────────────────────────────────
        if st.session_state.sec == "Verdict":
            svs = r.get("sub_verdicts", [])
            if len(svs) > 1:
                st.markdown('<div class="sub-lbl">Sub-claim breakdown</div>', unsafe_allow_html=True)
            for sv in svs:
                svc = VC.get(sv["verdict"], "nei")
                pct = int(sv["confidence_score"] * 100)
                st.markdown(
                    f'<div class="verdict-card vc-{svc}">'
                    f'<div class="vc-label">{sv["verdict"].upper()}</div>'
                    f'<div class="vc-text">{sv["sub_claim"]}</div>'
                    f'<div class="conf-track">'
                    f'<div class="conf-fill cf-{svc}" style="width:{pct}%"></div></div>'
                    f'<span class="conf-pct">{pct}% confidence</span>'
                    f'</div>',
                    unsafe_allow_html=True)

        # ── EVIDENCE ─────────────────────────────────────────────────────
        elif st.session_state.sec == "Evidence":
            try:    passages = get_evidence(r.get("claim_id", ""))
            except: passages = []

            if not passages:
                for sv in r.get("sub_verdicts", []):
                    for ev in sv.get("supporting_evidence", []):
                        passages.append({
                            "source": "—", "pmid": "", "title": ev.get("titolo", ""),
                            "journal": "", "year": "",
                            "testo": ev.get("testo") or ev.get("text", ""),
                            "url": ev.get("url", ""), "verdict_direction": "neutral"
                        })

            st.markdown(
                f'<p style="font-size:12px;color:#475569;margin-bottom:12px;">'
                f'{len(passages)} passages retrieved</p>',
                unsafe_allow_html=True)

            DD = {"supports": "dd-sup", "refutes": "dd-ref", "neutral": "dd-neu"}
            for p in passages:
                dc   = DD.get(p.get("verdict_direction", "neutral"), "dd-neu")
                link = (f'<a class="pmid-lnk" href="{p["url"]}" target="_blank">PMID {p["pmid"]}</a>'
                        if p.get("pmid") and p.get("url") else "")
                txt  = p.get("testo") or p.get("text") or p.get("abstract", "")
                st.markdown(
                    f'<div class="ev-card">'
                    f'<div style="display:flex;justify-content:space-between;gap:8px;margin-bottom:6px;">'
                    f'<div class="ev-title">{p.get("title","—")}</div>'
                    f'<div class="dir-dot {dc}"></div></div>'
                    f'<div class="ev-meta">'
                    f'<span class="badge b-src">{p.get("source","")}</span>'
                    f'{"<span>"+p["journal"]+"</span>" if p.get("journal") else ""}'
                    f'{"<span>"+str(p["year"])+"</span>" if p.get("year") else ""}'
                    f'{link}</div>'
                    f'<div class="ev-text">{txt}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # ── REASONING ────────────────────────────────────────────────────
        elif st.session_state.sec == "Reasoning":
            svs = r.get("sub_verdicts", [])
            for i, sv in enumerate(svs, 1):
                if len(svs) > 1:
                    st.markdown(
                        f'<div class="sub-lbl">Sub-claim {i}</div>'
                        f'<div class="sub-text">{sv["sub_claim"]}</div>',
                        unsafe_allow_html=True)
                cot = sv.get("chain_of_thought_log", "").replace("<", "&lt;")
                st.markdown(f'<div class="cot-box">{cot}</div>', unsafe_allow_html=True)

        # ── AGENT TRACE ───────────────────────────────────────────────────
        elif st.session_state.sec == "Agent trace":
            trace = r.get("agent_trace", [])
            if not trace:
                st.info("Agent trace not available for this claim.")
            else:
                st.markdown('<div class="trace-wrap">', unsafe_allow_html=True)
                for step in trace:
                    
                    # Caso 1: Array di stringhe testuali (provenienti dal tuo MultiAgentState log)
                    if isinstance(step, str):
                        agent_name = "System"
                        msg = step
                        if step.startswith("[") and "]: " in step:
                            agent_name, msg = step[1:].split("]: ", 1)
                            
                        st.markdown(
                            f'<div class="trace-step"><div class="trace-dot"></div>'
                            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
                            f'<div class="trace-agent">{agent_name}</div>'
                            f'</div>'
                            f'<div class="trace-model">{msg}</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                            
                    # Caso 2: Array di dizionari strutturati (compatibilità originaria dashboard)
                    elif isinstance(step, dict):
                        chips = ""
                        if step.get("n_sub_claims") is not None:
                            chips += f'<span class="trace-chip">⚡ {step["n_sub_claims"]} sub-claim</span>'
                        if step.get("n_passages") is not None:
                            chips += f'<span class="trace-chip">📄 {step["n_passages"]} passage</span>'
                        if step.get("verdict"):
                            chips += f'<span class="trace-chip">→ {step["verdict"]}</span>'
                        ok_col = "#4ade80" if step.get("status") == "ok" else "#f87171"
                        st.markdown(
                            f'<div class="trace-step"><div class="trace-dot"></div>'
                            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
                            f'<div class="trace-agent">{step.get("agent","")}</div>'
                            f'<span style="font-size:11px;color:{ok_col};">● {step.get("status","ok")}</span>'
                            f'</div>'
                            f'<div class="trace-model">{step.get("model","")}</div>'
                            f'<div style="margin-top:4px;">{chips}</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)