
from pathlib import Path
import html
import json
import os

import streamlit as st

# ---------------------------------------------------------
# PROJECT PATHS — KEEP PROJECT ASSETS ON D:
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
STORAGE_DIR = ROOT / "storage"
CACHE_DIR = ROOT / ".cache" / "huggingface"
TEMP_DIR = ROOT / ".temp"

for directory in (RAW_DIR, STORAGE_DIR, CACHE_DIR, TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)

# Ensure relative paths used by the existing project modules
# resolve to this project's directory.
os.chdir(ROOT)

from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.retriever import PolicyRetriever
from src.rag_pipeline import PolicyRAG


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Policy Intelligence | Knowledge Platform",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# PROFESSIONAL UI
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --navy: #172B4D;
        --teal: #147D78;
        --ink: #253247;
        --muted: #6B778C;
        --line: #E1E7EF;
        --canvas: #F5F7FA;
        --white: #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245,247,250,.95);
    }

    [data-testid="stSidebar"] {
        background: var(--white);
        border-right: 1px solid var(--line);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 0 22px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 22px;
    }

    .brand-mark {
        width: 46px;
        height: 46px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: var(--navy);
        color: white;
        font: 800 15px 'Manrope', sans-serif;
    }

    .brand-name {
        color: var(--navy);
        font: 800 15px 'Manrope', sans-serif;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 11px;
        margin-top: 4px;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: var(--navy);
        font: 800 clamp(29px, 3.5vw, 43px)/1.2 'Manrope', sans-serif;
        letter-spacing: -1.5px;
        margin: 0 0 14px;
    }

    .hero-description {
        max-width: 760px;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.9;
        margin-bottom: 28px;
    }

    .section-heading {
        font: 800 17px 'Manrope', sans-serif;
        color: var(--navy);
        margin: 26px 0 12px;
    }

    .section-caption {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 12px;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 17px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 7px;
        min-height: 42px;
        font-weight: 700;
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--navy);
        border-color: var(--navy);
        color: white;
    }

    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: var(--teal);
        border-color: var(--teal);
        color: white;
    }

    .stTextInput input {
        min-height: 45px;
        border-radius: 7px;
    }

    [data-testid="stFileUploader"] {
        background: #FBFCFE;
        border: 1px dashed #C7D3E0;
        border-radius: 9px;
        padding: 8px;
    }

    .answer-heading {
        color: var(--teal);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .answer-body {
        color: var(--ink);
        font-size: 15px;
        line-height: 1.9;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .source-card {
        background: #F8FAFC;
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 7px;
        padding: 14px 16px;
        margin: 10px 0;
    }

    .source-title {
        color: var(--navy);
        font-size: 13px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .source-detail {
        color: var(--muted);
        font-size: 12px;
        margin-top: 5px;
    }

    .welcome-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 24px;
        margin-top: 18px;
    }

    .welcome-title {
        color: var(--navy);
        font: 800 19px 'Manrope', sans-serif;
        margin-bottom: 8px;
    }

    .welcome-copy {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.9;
    }

    .footer {
        margin-top: 60px;
        padding: 30px 12px 25px;
        border-top: 2px solid var(--teal);
        text-align: center;
    }

    .footer-label {
        color: var(--teal);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .footer-name {
        color: var(--navy);
        font: 800 clamp(29px, 4vw, 41px)/1.3 'Manrope', sans-serif;
        letter-spacing: -1.2px;
        overflow-wrap: anywhere;
    }

    .footer-role {
        color: var(--muted);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.3px;
        margin-top: 10px;
    }

    .footer-divider {
        color: var(--teal);
        padding: 0 7px;
    }

    .footer-product {
        color: #8792A2;
        font-size: 12px;
        margin-top: 14px;
    }

    @media (max-width: 700px) {
        .hero-title {
            font-size: 29px;
        }

        .footer-role {
            font-size: 11px;
            letter-spacing: .6px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DOCUMENT AND INDEX HELPERS
# ---------------------------------------------------------

def saved_pdfs() -> list[str]:
    return sorted(
        p.name for p in RAW_DIR.glob("*.pdf") if p.is_file()
    )


def saved_chunk_count() -> int:
    metadata = STORAGE_DIR / "chunks.json"

    if not metadata.exists():
        return 0

    try:
        with metadata.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        return len(chunks)
    except (OSError, ValueError):
        return 0


def load_pipeline():
    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = PolicyRAG(
                storage_folder=STORAGE_DIR,
                min_similarity=0.35,
            )
            st.session_state.pipeline_error = None
        except Exception as exc:
            st.session_state.pipeline = None
            st.session_state.pipeline_error = str(exc)

    return st.session_state.pipeline


def rebuild_index() -> int:
    documents = load_documents(RAW_DIR)

    if not documents:
        raise ValueError(
            "No readable PDF documents were found in data/raw."
        )

    chunks = chunk_documents(
        documents,
        chunk_size=700,
        overlap=1,
    )

    if not chunks:
        raise ValueError(
            "No searchable text was extracted from the PDFs."
        )

    retriever = PolicyRetriever(cache_folder=CACHE_DIR)
    retriever.build_index(chunks)
    retriever.save_index(STORAGE_DIR)

    # Reload from the newly saved index.
    st.session_state.pipeline = PolicyRAG(
        storage_folder=STORAGE_DIR,
        min_similarity=0.35,
    )
    st.session_state.pipeline_error = None

    return len(chunks)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">PI</div>
            <div>
                <div class="brand-name">Policy Intelligence</div>
                <div class="brand-caption">
                    Knowledge &amp; Compliance
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Document library")
    st.markdown(
        '<div class="section-caption">'
        'Upload policy PDFs to make their contents searchable.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploads = st.file_uploader(
        "Select PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Documents are saved locally in data/raw.",
    )

    if st.button(
        "Upload and index PDFs",
        type="primary",
        use_container_width=True,
    ):
        if not uploads:
            st.warning("Select at least one PDF.")
        else:
            try:
                for uploaded in uploads:
                    filename = Path(uploaded.name).name

                    if not filename.lower().endswith(".pdf"):
                        raise ValueError("Only PDF files are supported.")

                    (RAW_DIR / filename).write_bytes(
                        uploaded.getvalue()
                    )

                with st.spinner(
                    "Extracting text and building semantic search..."
                ):
                    count = rebuild_index()

                st.success(
                    f"Index ready: {count} searchable sections."
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    if st.button(
        "Rebuild from saved PDFs",
        use_container_width=True,
    ):
        try:
            with st.spinner("Rebuilding the policy index..."):
                count = rebuild_index()

            st.success(f"Index rebuilt with {count} sections.")
            st.rerun()

        except Exception as exc:
            st.error(f"Could not rebuild the index: {exc}")

    st.markdown("---")
    st.markdown("**Available documents**")

    filenames = saved_pdfs()

    if filenames:
        for filename in filenames:
            st.markdown(
                f"📄 {html.escape(filename)}",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No PDFs found in data/raw.")

    st.markdown("---")
    st.caption(
        "Local project storage · Semantic retrieval · "
        "Evidence-based policy search"
    )


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="eyebrow">Policy Intelligence Platform</div>

    <div class="hero-title">
        Turn policy documents into trusted answers.
    </div>

    <div class="hero-description">
        Search organizational policies, retrieve relevant evidence,
        and verify every answer against its original document and
        page reference.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# STATUS METRICS
# ---------------------------------------------------------

pipeline = load_pipeline()
filenames = saved_pdfs()
chunk_count = saved_chunk_count()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Policy documents", len(filenames))

with col2:
    st.metric("Indexed sections", chunk_count)

with col3:
    st.metric(
        "Search engine",
        "Ready" if pipeline is not None else "Not ready",
    )

if pipeline is None:
    st.warning(
        "The search index could not be loaded. "
        "Use 'Rebuild from saved PDFs' in the sidebar if your "
        "PDF documents are already present."
    )

    error_detail = st.session_state.get("pipeline_error")
    if error_detail:
        with st.expander("Technical details"):
            st.code(error_detail)


# ---------------------------------------------------------
# POLICY QUESTION FORM
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">Ask your policy question</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-caption">'
    'Ask a specific question. Results include supporting passages '
    'from the available documents.'
    '</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    with st.form("policy_search_form", clear_on_submit=False):
        question = st.text_input(
            "Question",
            placeholder=(
                "Example: What should I do if I arrive late?"
            ),
            label_visibility="collapsed",
        )

        search_clicked = st.form_submit_button(
            "Search policy documents",
            type="primary",
            use_container_width=True,
        )

if search_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif pipeline is None:
        st.error(
            "Search is unavailable. Rebuild the index from the sidebar."
        )
    else:
        try:
            with st.spinner("Finding relevant policy evidence..."):
                st.session_state.last_result = pipeline.ask(
                    question.strip(),
                    top_k=3,
                )
        except Exception as exc:
            st.error(f"Search failed: {exc}")


# ---------------------------------------------------------
# RESULTS AND SOURCES
# ---------------------------------------------------------

result = st.session_state.get("last_result")

if result:
    st.markdown(
        '<div class="section-heading">Answer and evidence</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<div class="answer-heading">Policy response</div>',
            unsafe_allow_html=True,
        )

        if result.get("supported"):
            answer = html.escape(str(result.get("answer", "")))

            st.markdown(
                f'<div class="answer-body">{answer}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(result.get("answer", "No supported answer found."))

        st.markdown("---")
        st.markdown("**Source references**")

        sources = result.get("sources", [])

        if sources:
            for source in sources:
                doc = html.escape(str(source.get("document", "Unknown")))
                page = source.get("page", "?")
                similarity = source.get("similarity", 0.0)

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">{doc}</div>
                        <div class="source-detail">
                            Page {page}
                            &nbsp;·&nbsp;
                            Retrieval similarity: {similarity:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(
                    f"Read evidence — {source.get('document', 'Source')}, "
                    f"page {page}"
                ):
                    st.write(source.get("evidence", ""))

        else:
            st.info(
                "No sufficiently relevant passages were found. "
                "Consult the policy owner or add the relevant document."
            )

elif pipeline is not None:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-title">Your policy workspace is ready.</div>
            <div class="welcome-copy">
                Enter a question above to search your indexed documents.
                You can ask about attendance, leave approval, remote work,
                password security, or any other subject covered by your PDFs.
                Answers are grounded in retrieved passages, with source
                references shown for verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# DEVELOPER SIGNATURE
# ---------------------------------------------------------


st.markdown(
    """
    <div class="custom-footer">
        <div class="footer-label">Developed By</div>
        <div class="footer-name">Muzamil Rasul</div>
        <div class="footer-role">
            DATA SCIENTIST
            <span class="footer-divider">|</span>
            ML ENGINEER
        </div>
        <div class="footer-product">
            Policy Intelligence · Knowledge &amp; Compliance
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
