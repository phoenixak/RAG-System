"""
Global Styles
Dark & Technical aesthetic CSS for the Enterprise RAG System frontend.
Inject via apply_global_styles() at app startup.
"""

import streamlit as st


GOOGLE_FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
"""

GLOBAL_CSS = """
<style>
/* ============================================================
   DESIGN TOKENS
   ============================================================ */
:root {
    --bg-base:        #080c16;
    --bg-surface:     #0f1422;
    --bg-elevated:    #161c2e;
    --bg-card:        #1a2133;
    --bg-input:       #111827;

    --accent-cyan:    #00d4ff;
    --accent-purple:  #7c3aed;
    --accent-green:   #10b981;
    --accent-amber:   #f59e0b;
    --accent-red:     #ef4444;

    --text-primary:   #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;
    --text-accent:    #00d4ff;

    --border-subtle:  rgba(0, 212, 255, 0.08);
    --border-normal:  rgba(0, 212, 255, 0.18);
    --border-strong:  rgba(0, 212, 255, 0.4);

    --glow-cyan:      0 0 20px rgba(0, 212, 255, 0.25);
    --glow-purple:    0 0 20px rgba(124, 58, 237, 0.25);

    --font-display:   'Syne', sans-serif;
    --font-body:      'Inter', sans-serif;
    --font-mono:      'DM Mono', monospace;

    --radius-sm:      4px;
    --radius-md:      8px;
    --radius-lg:      12px;
}

/* ============================================================
   BASE RESET & LAYOUT
   ============================================================ */
.stApp {
    background-color: var(--bg-base) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px !important;
}

/* ============================================================
   TYPOGRAPHY
   ============================================================ */
h1 {
    font-family: var(--font-display) !important;
    font-weight: 800 !important;
    font-size: 1.9rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}

h1::before {
    content: '// ';
    color: var(--accent-cyan);
    font-family: var(--font-mono);
    font-size: 0.85em;
    opacity: 0.7;
}

h2 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em !important;
}

h3 {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    color: var(--text-secondary) !important;
}

p, .stMarkdown p {
    font-family: var(--font-body) !important;
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}

/* ============================================================
   SIDEBAR
   ============================================================ */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-family: var(--font-body) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 0.5rem 0.8rem !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(0, 212, 255, 0.07) !important;
    border-color: var(--border-normal) !important;
    color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* Sidebar metrics */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.6rem 0.8rem !important;
    margin-bottom: 0.4rem !important;
}

[data-testid="stSidebar"] [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.3rem !important;
    color: var(--accent-cyan) !important;
}

[data-testid="stSidebar"] [data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ============================================================
   MAIN BUTTONS
   ============================================================ */
.stButton button {
    background: transparent !important;
    border: 1px solid var(--border-normal) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.83rem !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.02em !important;
}

.stButton button:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

.stButton button[kind="primary"] {
    background: rgba(124, 58, 237, 0.2) !important;
    border-color: var(--accent-purple) !important;
    color: #c4b5fd !important;
}

.stButton button[kind="primary"]:hover {
    background: rgba(124, 58, 237, 0.35) !important;
    box-shadow: var(--glow-purple) !important;
    color: #e9d5ff !important;
}

/* ============================================================
   INPUTS & FORMS
   ============================================================ */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select,
.stNumberInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.18s ease !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.2) !important;
    outline: none !important;
}

.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stSlider label,
.stNumberInput label,
.stCheckbox label {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
}

/* Selectbox dropdown */
[data-baseweb="select"] > div {
    background: var(--bg-input) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}

[data-baseweb="select"] span {
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-cyan) !important;
    border-color: var(--accent-cyan) !important;
}

/* ============================================================
   CHAT MESSAGES
   ============================================================ */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--border-normal) !important;
}

/* User message */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--accent-cyan) !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--accent-purple) !important;
    background: rgba(124, 58, 237, 0.04) !important;
}

[data-testid="stChatMessage"] p {
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    font-family: var(--font-body) !important;
}

/* Chat input bar */
[data-testid="stChatInput"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-normal) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--glow-cyan) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic !important;
}

/* ============================================================
   EXPANDERS
   ============================================================ */
[data-testid="stExpander"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 0.8rem !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--accent-cyan) !important;
}

/* ============================================================
   METRICS
   ============================================================ */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.8rem 1rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
    color: var(--accent-cyan) !important;
    font-weight: 500 !important;
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
}

/* ============================================================
   ALERTS / INFO BOXES
   ============================================================ */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: none !important;
    font-family: var(--font-body) !important;
    font-size: 0.87rem !important;
}

.stSuccess {
    background: rgba(16, 185, 129, 0.1) !important;
    border-left: 3px solid var(--accent-green) !important;
    color: #6ee7b7 !important;
}

.stError {
    background: rgba(239, 68, 68, 0.1) !important;
    border-left: 3px solid var(--accent-red) !important;
    color: #fca5a5 !important;
}

.stWarning {
    background: rgba(245, 158, 11, 0.1) !important;
    border-left: 3px solid var(--accent-amber) !important;
    color: #fcd34d !important;
}

.stInfo {
    background: rgba(0, 212, 255, 0.06) !important;
    border-left: 3px solid var(--accent-cyan) !important;
    color: var(--text-secondary) !important;
}

/* ============================================================
   DIVIDERS
   ============================================================ */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.2rem 0 !important;
}

/* ============================================================
   TABS
   ============================================================ */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border-subtle) !important;
    gap: 0 !important;
}

[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.1rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.18s ease !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
    background: transparent !important;
}

[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

/* ============================================================
   SPINNERS & PROGRESS
   ============================================================ */
[data-testid="stSpinner"] {
    color: var(--accent-cyan) !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan)) !important;
    border-radius: 9999px !important;
}

.stProgress > div {
    background: var(--bg-elevated) !important;
    border-radius: 9999px !important;
    height: 4px !important;
}

/* ============================================================
   CAPTIONS & SMALL TEXT
   ============================================================ */
.stCaption, small, caption {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.02em !important;
}

/* ============================================================
   CONTAINERS / CARDS
   ============================================================ */
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
    gap: 0.75rem !important;
}

/* ============================================================
   SCROLLBAR
   ============================================================ */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: var(--bg-base);
}
::-webkit-scrollbar-thumb {
    background: var(--border-normal);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-cyan);
}

/* ============================================================
   LOGIN PAGE SPECIFIC
   ============================================================ */
.rag-login-hero {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.06) 0%, rgba(124, 58, 237, 0.06) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.rag-login-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0, 212, 255, 0.04) 0%, transparent 60%);
    pointer-events: none;
}

.rag-prompt {
    font-family: var(--font-mono);
    color: var(--accent-cyan);
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
    letter-spacing: 0.04em;
    opacity: 0.8;
}

.rag-title {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.rag-title span {
    color: var(--accent-cyan);
}

.rag-subtitle {
    font-family: var(--font-body);
    font-size: 0.9rem;
    color: var(--text-muted);
    line-height: 1.6;
    max-width: 500px;
}

.rag-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    border: 1px solid var(--border-normal);
    color: var(--accent-cyan);
    background: rgba(0, 212, 255, 0.07);
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    letter-spacing: 0.04em;
}

.rag-status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    padding: 0.6rem 0.9rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
    background: var(--bg-elevated);
}

.rag-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
    animation: pulse-dot 2s ease-in-out infinite;
}

.rag-status-dot.offline {
    background: var(--accent-red);
    box-shadow: 0 0 6px var(--accent-red);
    animation: none;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ============================================================
   CHAT PAGE SPECIFIC
   ============================================================ */
.chat-page-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.25rem;
}

.chat-page-tag {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.chat-controls-bar {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
    padding: 0.6rem 0.9rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin-bottom: 1rem;
}

.search-result-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-cyan);
    border-radius: var(--radius-md);
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.87rem;
}

.search-result-card .source-name {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--accent-cyan);
    font-weight: 500;
    margin-bottom: 0.4rem;
}

.search-result-card .score-badge {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    float: right;
}

.search-result-card .content-text {
    color: var(--text-secondary);
    line-height: 1.55;
}

/* ============================================================
   FILE UPLOADER
   ============================================================ */
[data-testid="stFileUploader"] {
    background: var(--bg-elevated) !important;
    border: 1px dashed var(--border-normal) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(0, 212, 255, 0.03) !important;
}

/* ============================================================
   HIDE STREAMLIT BRANDING
   ============================================================ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
"""


def apply_global_styles() -> None:
    """Inject global fonts and CSS into the Streamlit app."""
    st.markdown(GOOGLE_FONTS, unsafe_allow_html=True)
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_login_hero() -> None:
    """Render the styled hero section for the login page."""
    st.markdown(
        """
        <div class="rag-login-hero">
            <div class="rag-prompt">$ enterprise-rag --version 1.0.0 --env production</div>
            <div class="rag-title">Enterprise <span>RAG</span><br>System</div>
            <div class="rag-subtitle">
                Intelligent document search and conversational AI.
                Semantic, hybrid &amp; contextual retrieval over your private knowledge base.
            </div>
            <br>
            <span class="rag-badge">FastAPI</span>
            <span class="rag-badge">ChromaDB</span>
            <span class="rag-badge">Sentence Transformers</span>
            <span class="rag-badge">LLM-powered</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_indicator(online: bool, label: str = "", detail: str = "") -> None:
    """Render a compact online/offline status indicator."""
    dot_class = "rag-status-dot" if online else "rag-status-dot offline"
    status_text = label if label else ("Online" if online else "Offline")
    detail_html = (
        f" &mdash; <span style='color:var(--text-muted)'>{detail}</span>"
        if detail
        else ""
    )
    st.markdown(
        f"""
        <div class="rag-status-indicator">
            <span class="{dot_class}"></span>
            <span style="color:var(--text-secondary); font-family:var(--font-mono); font-size:0.8rem;">
                {status_text}{detail_html}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
