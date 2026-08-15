import streamlit as st
import base64
import html
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PaperLense",
    page_icon="P",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

BACKGROUND_PATH = BASE_DIR / "assets" / "background.png"
CSS_PATH = BASE_DIR / "style.css"

CSV_PATH = BASE_DIR / "data" / "ML-ArXiv-Papers.csv"
FAISS_PATH = BASE_DIR / "models" / "paper_faiss.index"


# ============================================================
# LOAD EXISTING UI CSS
# ============================================================
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.html(f"<style>{f.read()}</style>")


# ============================================================
# BACKGROUND IMAGE
# ============================================================
if BACKGROUND_PATH.exists():
    with open(BACKGROUND_PATH, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    st.html(f"""
    <style>
    html, body {{
        margin: 0 !important;
        padding: 0 !important;
    }}

    .stApp {{
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;

        background-image:
            linear-gradient(
                rgba(255, 255, 255, 0.82),
                rgba(255, 255, 255, 0.82)
            ),
            url("data:image/png;base64,{image_base64}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;

        z-index: -10;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    .main,
    .block-container {{
        background: transparent !important;
    }}
    </style>
    """)


# ============================================================
# REAL SEARCH BACKEND
# ============================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_index():
    return faiss.read_index(str(FAISS_PATH))


@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)


# ============================================================
# DATA HELPERS
# ============================================================
def find_column(df, candidates):
    exact = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]

    for col in df.columns:
        for candidate in candidates:
            if candidate.lower() in str(col).lower():
                return col

    return None


def get_value(row, column, fallback=""):
    if column is not None:
        value = row[column]

        if pd.isna(value):
            return ""

        return str(value)

    return fallback


# ============================================================
# SUMMARY
# ============================================================
def make_summary(text, length):
    text = " ".join(str(text).split())

    if not text:
        return "No abstract available."

    sentences = [
        s.strip()
        for s in text
        .replace("!", ".")
        .replace("?", ".")
        .split(".")
        if s.strip()
    ]

    summary = ". ".join(sentences[:2])

    limit = max(300, length * 10)

    if len(summary) > limit:
        summary = summary[:limit].rsplit(" ", 1)[0] + "..."

    if summary and not summary.endswith("."):
        summary += "."

    return summary


# ============================================================
# KEYWORDS
# ============================================================
def extract_keywords(text, keyword_range):

    stopwords = {
        "the", "and", "for", "with", "that", "this",
        "from", "are", "was", "were", "have", "has",
        "had", "using", "into", "their", "there",
        "these", "they", "which", "while", "than",
        "also", "more", "most", "such", "paper",
        "method", "methods", "approach", "based",
        "proposed", "results", "shown", "present",
        "presented", "can", "may", "our", "its",
        "not", "between", "through", "within",
        "about", "over", "under", "new", "used"
    }

    words = []

    for raw in str(text).split():

        word = "".join(
            ch for ch in raw
            if ch.isalnum() or ch == "-"
        ).strip("-").lower()

        if (
            len(word) >= 4
            and word not in stopwords
            and not word.isdigit()
        ):
            words.append(word)

    words = list(dict.fromkeys(words))

    if keyword_range == "1–2 words":
        return words[:6]

    if keyword_range == "Single words":
        return words[:5]

    return words[:8]


# ============================================================
# SEMANTIC SEARCH
# ============================================================
@st.cache_data
def semantic_search(query, top_k):

    model = load_model()
    index = load_index()
    df = load_data()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    title_col = find_column(
        df,
        [
            "title",
            "paper_title",
            "paper title",
            "name"
        ]
    )

    abstract_col = find_column(
        df,
        [
            "abstract",
            "paper_text",
            "paper text",
            "text",
            "summary"
        ]
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx]

        title = get_value(
            row,
            title_col,
            "Untitled Research Paper"
        )

        abstract = get_value(
            row,
            abstract_col,
            ""
        )

        score = max(
            0.0,
            min(
                100.0,
                float(distance) * 100.0
            )
        )

        results.append({
            "title": title,
            "score": round(score, 1),
            "abstract": abstract
        })

    return results


# ============================================================
# SESSION STATE
# ============================================================
if "searched" not in st.session_state:
    st.session_state.searched = False

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


# ============================================================
# EXAMPLE BUTTON CALLBACK
# IMPORTANT:
# This avoids Streamlit's session_state widget error.
# ============================================================
def use_example(example):
    st.session_state.query_input = example
    st.session_state.searched = False


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.html("""
    <div class="sidebar-brand">
        <div class="brand-mark">P</div>

        <div>
            <div class="brand-name">PaperLense</div>
            <div class="brand-subtitle">
                Research Assistant
            </div>
        </div>
    </div>

    <div class="sidebar-heading">
        FILTERS
    </div>
    """)

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    summary_length = st.slider(
        "Summary length",
        min_value=40,
        max_value=120,
        value=80,
        step=10
    )

    keyword_range = st.selectbox(
        "Keyword range",
        [
            "1–3 words",
            "1–2 words",
            "Single words"
        ]
    )

    st.write("")

    if st.button(
        "↻  Clear Search",
        use_container_width=True
    ):
        st.session_state.searched = False
        st.session_state.query_input = ""
        st.rerun()

    st.html("""
    <div class="sidebar-footer">
        <div>PaperLense v1.0</div>
        <div>AI Research Engine</div>
    </div>
    """)

# ============================================================
# TOP NAVBAR
# ============================================================
st.html("""
<div class="top-navbar">

    <div class="nav-brand">
        PaperLense
    </div>

    <div class="nav-links">

        <span class="active-nav">
            Research Assistant
        </span>

        <span>
            About
        </span>

        <span>
            History
        </span>

    </div>

</div>
""")


# ============================================================
# HERO
# ============================================================
if not st.session_state.searched:

    st.html("""
    <div class="hero">

        <div class="hero-badge">
            <span class="status-dot"></span>
            AI-POWERED RESEARCH ENGINE
        </div>

        <h1>
            Find the right research papers.
            <br>
            <span>Faster.</span>
        </h1>

        <p>
            Search Machine Learning research papers using
            natural language and instantly get relevant
            papers, summaries and key insights.
        </p>

    </div>
    """)


# ============================================================
# SEARCH BOX
# ============================================================
st.html(
    '<div class="search-wrapper"></div>'
)

query = st.text_input(
    "Research question",
    key="query_input",
    placeholder="Ask your research question...",
    label_visibility="collapsed"
)

search_clicked = st.button(
    "Search →",
    use_container_width=True
)


# ============================================================
# SEARCH BUTTON
# ============================================================
if search_clicked:

    if query.strip():

        st.session_state.searched = True

        st.rerun()

    else:

        st.warning(
            "Please enter a research question first."
        )


# ============================================================
# EXAMPLES
# ============================================================
if not st.session_state.searched:

    st.html(
        '<div class="example-title">Try an example</div>'
    )

    example1, example2, example3 = st.columns(3)

    examples = [
        (
            "How does attention improve image classification?",
            example1
        ),
        (
            "Recent approaches for object detection",
            example2
        ),
        (
            "Transformer models for medical imaging",
            example3
        )
    ]

    for i, (example, column) in enumerate(examples):

        with column:

            st.button(
                example,
                key=f"example_{i}",
                use_container_width=True,
                on_click=use_example,
                args=(example,)
            )


# ============================================================
# RESULTS
# ============================================================
if st.session_state.searched:

    current_query = st.session_state.query_input.strip()

    if not current_query:

        st.session_state.searched = False
        st.rerun()

    try:

        with st.spinner(
            "Searching research papers semantically..."
        ):

            selected_papers = semantic_search(
                current_query,
                top_k
            )

    except Exception as e:

        st.error(
            "Semantic search could not be completed."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # RESULTS HEADER
    # --------------------------------------------------------
    st.html(f"""
    <div class="results-header">

        <div>

            <div class="results-label">
                SEARCH RESULTS
            </div>

            <h2>
                Results for
                <span>
                    "{html.escape(current_query)}"
                </span>
            </h2>

        </div>

        <div class="results-count">
            {len(selected_papers)} papers found
        </div>

    </div>
    """)


    # --------------------------------------------------------
    # PAPER CARDS
    # --------------------------------------------------------
    for index, paper in enumerate(selected_papers):

        score = paper["score"]

        keywords = extract_keywords(
            paper["abstract"],
            keyword_range
        )

        keywords_html = "".join(
            f'<span class="keyword">'
            f'{html.escape(k)}'
            f'</span>'
            for k in keywords
        )

        summary = make_summary(
            paper["abstract"],
            summary_length
        )


        st.html(f"""
        <div class="paper-card">

            <div class="paper-top">

                <div class="paper-number">
                    {index + 1:02d}
                </div>

                <div class="paper-main">

                    <h3>
                        {html.escape(paper["title"])}
                    </h3>

                    <div class="similarity-row">

                        <span class="similarity-label">
                            Similarity
                        </span>

                        <span class="similarity-score">
                            {score}%
                        </span>

                    </div>

                    <div class="score-bar">

                        <div
                            class="score-fill"
                            style="width: {score}%;">
                        </div>

                    </div>

                </div>

            </div>


            <div class="section-label">
                AI-GENERATED SUMMARY
            </div>

            <p class="summary-text">
                {html.escape(summary)}
            </p>


            <div class="section-label">
                KEY PHRASES
            </div>

            <div class="keyword-container">
                {keywords_html}
            </div>

        </div>
        """)


        # ----------------------------------------------------
        # DETAILS BUTTON
        # ----------------------------------------------------
        if st.button(
            "View paper details →",
            key=f"details_{index}"
        ):

            st.html(f"""
            <div class="detail-box">

                <div class="detail-title">
                    {html.escape(paper["title"])}
                </div>

                <div class="detail-score">
                    Relevance Score:
                    <strong>{score}%</strong>
                </div>

                <div class="detail-heading">
                    ABSTRACT
                </div>

                <p>
                    {html.escape(paper["abstract"])}
                </p>

                <div class="detail-heading">
                    AI SUMMARY
                </div>

                <p>
                    {html.escape(summary)}
                </p>

                <div class="detail-heading">
                    KEY PHRASES
                </div>

                <div class="keyword-container">
                    {keywords_html}
                </div>

            </div>
            """)


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------
    st.html("""
    <div class="results-footer">

        Results generated using semantic similarity,
        abstractive summarization and keyword extraction.

    </div>
    """)