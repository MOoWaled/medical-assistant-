"""Streamlit user interface for the Medical ChatBot project."""

from __future__ import annotations

import html
import os

import streamlit as st

from model_clients import (
    DEFAULT_BASELINE_URL,
    DEFAULT_MISTRAL_URL,
    DEFAULT_QWEN_URL,
    ModelRequestError,
    analyse_symptoms,
)


st.set_page_config(
    page_title="Medical ChatBot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_page_style() -> None:
    """Apply a calm, clinic-inspired visual system to the Streamlit UI."""
    st.markdown(
        """
        <style>
            :root {
                --ink: #133444;
                --muted: #617887;
                --teal: #168c97;
                --teal-dark: #0b6572;
                --blue: #1976c9;
                --sky: #e9f7f8;
                --surface: #ffffff;
                --line: #d8e8eb;
            }

            .stApp {
                background:
                    radial-gradient(circle at 92% 8%, rgba(75, 183, 194, .16) 0%, transparent 25rem),
                    radial-gradient(circle at 4% 42%, rgba(28, 130, 191, .10) 0%, transparent 24rem),
                    #f7fbfc;
                color: var(--ink);
            }
            .block-container {
                max-width: 1240px;
                padding: 1.45rem 2.4rem 4rem;
                animation: page-enter 500ms ease both;
            }
            [data-testid="stHeader"] {
                background: rgba(255, 255, 255, .86);
                border-bottom: 1px solid rgba(21, 85, 101, .10);
                backdrop-filter: blur(14px);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #f4fbfc 100%);
                border-right: 1px solid var(--line);
                box-shadow: 8px 0 30px rgba(20, 81, 95, .05);
            }
            [data-testid="stSidebar"] > div:first-child {
                padding: 1.75rem 1.25rem 2.5rem;
            }
            h1, h2, h3, p, label, .stCaption, [data-testid="stMarkdownContainer"] {
                color: var(--ink);
            }
            .hero {
                position:relative;
                overflow:hidden;
                margin:.5rem 0 1.2rem;
                padding:2.35rem 2.45rem;
                border:1px solid #d4e9ec;
                border-radius:1.3rem;
                background:linear-gradient(112deg, rgba(255,255,255,.98) 0%, rgba(235,249,250,.98) 64%, rgba(212,240,244,.92) 100%);
                box-shadow:0 14px 38px rgba(18,101,119,.09);
            }
            .hero::after { content:""; position:absolute; width:14rem; height:14rem; border:28px solid rgba(31,151,165,.10); border-radius:50%; right:-5rem; top:-8rem; }
            .hero-kicker { margin:0 0 .55rem; color:var(--teal-dark); font-size:.82rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; }
            .hero h1 { position:relative; z-index:1; margin:0 !important; font-size:clamp(2rem,4vw,3.35rem) !important; font-weight:800 !important; letter-spacing:-.04em; color:#123444 !important; }
            .hero h1 strong { color:var(--teal); }
            .hero p { position:relative; z-index:1; max-width:44rem; margin:.8rem 0 0; color:#55717c; font-size:1.02rem; line-height:1.65; }
            .trust-row { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.2rem; position:relative; z-index:1; }
            .trust-chip { border:1px solid #cce3e6; border-radius:99px; padding:.38rem .7rem; background:rgba(255,255,255,.72); color:#37616a; font-size:.8rem; font-weight:600; }
            .sidebar-title { margin:0 0 .25rem; font-size:1.25rem; font-weight:800; color:#173c4b; }
            .sidebar-copy { margin:0 0 1.5rem; color:#6b818a; font-size:.88rem; line-height:1.5; }
            [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
            [data-testid="stAlert"] {
                border-radius: .85rem;
                border: 1px solid #f0d99f;
                background: #fff9e9;
                box-shadow: none;
            }
            [data-testid="stAlert"] p { color: #725821 !important; }
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-baseweb="select"] > div,
            [data-testid="stNumberInput"] input {
                color: var(--ink) !important;
                background: #fff !important;
                border: 1px solid #cfe0e4 !important;
                border-radius: .7rem !important;
                box-shadow: 0 2px 7px rgba(20,91,104,.03);
            }
            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextArea"] textarea:focus,
            [data-baseweb="select"] > div:focus-within,
            [data-testid="stNumberInput"] input:focus {
                border-color: #48aeba !important;
                box-shadow: 0 0 0 3px rgba(45,158,171,.14) !important;
            }
            [data-testid="stTextArea"] textarea {
                min-height: 175px;
                line-height: 1.6;
            }
            [data-baseweb="select"] svg,
            [data-testid="stNumberInput"] button svg { fill: var(--teal-dark) !important; }
            [data-testid="stWidgetLabel"] p {
                color: #31535f !important;
                font-weight: 700;
                letter-spacing: 0.01em;
            }
            div.stButton > button {
                min-height: 3.15rem;
                border: 0;
                border-radius: .75rem;
                color: #fff;
                font-weight: 800;
                letter-spacing: 0.015em;
                background: linear-gradient(105deg, #1976c9, #168f99);
                box-shadow: 0 9px 18px rgba(22,132,151,.22);
                transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
            }
            div.stButton > button:hover {
                transform: translateY(-2px);
                filter: brightness(1.04);
                box-shadow: 0 13px 25px rgba(22,132,151,.27);
            }
            div.stButton > button:active { transform: translateY(0); }
            [data-testid="stDivider"] { border-color: var(--line); }
            .section-label { display:flex; align-items:center; gap:.55rem; margin:1.55rem 0 .65rem; color:#244d5c; font-weight:800; font-size:1.05rem; }
            .section-label span { display:grid; place-items:center; width:1.8rem; height:1.8rem; border-radius:.5rem; color:#fff; background:#168c97; }
            .result-card {
                position: relative;
                overflow: hidden;
                min-height: 135px;
                margin: 0.35rem 0 1rem;
                padding: 1.25rem 1.3rem;
                border: 1px solid #d7e7ea;
                border-radius: 1rem;
                background: rgba(255,255,255,.94);
                box-shadow: 0 8px 22px rgba(24,90,105,.07);
                animation: card-enter 420ms ease both;
                transition: transform 160ms ease, box-shadow 160ms ease;
            }
            .result-card::before {
                content: "";
                position: absolute;
                inset: 0 auto 0 0;
                width: 4px;
                background: linear-gradient(#1c79c7, #19a4a2);
            }
            .result-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 13px 27px rgba(24,90,105,.12);
            }
            .result-card h3 {
                margin: 0 0 0.7rem;
                color: #1c4d5a;
                font-size: 1rem;
                letter-spacing: 0.02em;
            }
            .result-card p {
                margin: 0;
                color: #58707b;
                line-height: 1.62;
                white-space: pre-wrap;
            }
            [data-testid="stExpander"] {
                border: 1px solid var(--line) !important;
                border-radius: .8rem !important;
                background: #fff;
            }
            [data-testid="stExpander"] summary { color: #31535f !important; }
            [data-testid="stJson"] { background: #eef6f8; border-radius: .6rem; }
            @keyframes page-enter {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes card-enter {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }
            }
            @media (max-width: 760px) {
                .block-container { padding: 1rem 1rem 3rem; }
                .hero { padding:1.6rem 1.35rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_result_card(title: str, value: str) -> None:
    """Render an escaped, palette-aware result card."""
    safe_title = html.escape(title)
    safe_value = html.escape(value or "No information returned by the selected model.")
    st.markdown(
        f'<section class="result-card"><h3>{safe_title}</h3><p>{safe_value}</p></section>',
        unsafe_allow_html=True,
    )


def render_branding() -> None:
    """Render the healthcare hero panel without a redundant navigation bar."""
    st.markdown(
        """
        <section class="hero">
          <p class="hero-kicker">Your local medical information assistant</p>
          <h1>Clearer symptom guidance, <strong>grounded in data.</strong></h1>
          <p>Describe your symptoms, choose a model, and receive structured information sourced only from the project's reviewed condition knowledge base.</p>
          <div class="trust-row"><span class="trust-chip">🔒 Privacy-aware</span><span class="trust-chip">✦ RAG grounded</span><span class="trust-chip">🩺 Educational support</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


add_page_style()
render_branding()
st.warning(
    "This tool provides educational information only; it is not a medical diagnosis. "
    "For severe symptoms or a medical emergency, contact local emergency services immediately."
)

with st.sidebar:
    local_models_only = os.getenv("LOCAL_MODELS_ONLY", "").lower() in {"1", "true", "yes"}
    sidebar_copy = (
        "Local machine-learning and deep-learning models only."
        if local_models_only
        else "Choose the model and optionally add patient context."
    )
    st.markdown(
        f'<p class="sidebar-title">🩺 Analysis settings</p><p class="sidebar-copy">{sidebar_copy}</p>',
        unsafe_allow_html=True,
    )
    available_models = ("Grounded FAISS Baseline", "Deep Learning (RNN)", "Deep Learning (LSTM)")
    if not local_models_only:
        available_models += ("Mistral", "Qwen")
    model_label = st.selectbox(
        "Choose a model",
        available_models,
        help="All choices return medical fields from the local condition knowledge base.",
    )

    baseline_url = st.text_input(
        "Grounded Baseline API URL",
        value=os.getenv("BASELINE_API_URL", DEFAULT_BASELINE_URL),
        help="The local Flask API from model_API/app.py, usually running on port 5000.",
    )
    qwen_url = ""
    mistral_url = ""
    mistral_api_key = ""
    if not local_models_only:
        qwen_url = st.text_input(
            "Qwen API URL",
            value=os.getenv("QWEN_API_URL", DEFAULT_QWEN_URL),
            help="Base ngrok/Ollama URL. The app calls its /api/chat endpoint.",
        )
        mistral_url = st.text_input(
            "Mistral API URL",
            value=os.getenv("MISTRAL_API_URL", DEFAULT_MISTRAL_URL),
            help="Custom ngrok/Flask base URL (the app calls /generate), or https://api.mistral.ai/v1.",
        )
        mistral_api_key = st.text_input(
            "Mistral API key (optional)",
            value=os.getenv("MISTRAL_API_KEY", ""),
            type="password",
        )

    st.divider()
    st.subheader("Optional patient context")
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=None,
        step=1,
        help="Optional context. It never adds medical facts outside the knowledge base.",
    )
    gender_choice = st.selectbox(
        "Gender",
        ("Not specified", "Female", "Male"),
        help="Optional context used only when choosing among retrieved conditions.",
    )
    gender = None if gender_choice == "Not specified" else gender_choice

    st.divider()
    st.caption("Endpoint addresses are used only for the current request and are not saved by the app.")

st.markdown('<p class="section-label"><span>✦</span> Tell us what you are experiencing</p>', unsafe_allow_html=True)
symptoms = st.text_area(
    "Symptoms or medical concern",
    placeholder="Example: I have a persistent cough, wheezing, and shortness of breath.",
    height=190,
)

submit = st.button("Analyse symptoms", type="primary", use_container_width=True)

if submit:
    if not symptoms.strip():
        st.error("Please describe at least one symptom before continuing.")
        st.stop()

    selected_model = {
        "Grounded FAISS Baseline": "logistic_baseline",
        "Deep Learning (RNN)": "deep_rnn",
        "Deep Learning (LSTM)": "deep_lstm",
        "Mistral": "mistral",
        "Qwen": "qwen",
    }[model_label]

    with st.spinner(f"Contacting {model_label}..."):
        try:
            result = analyse_symptoms(
                model=selected_model,
                symptoms=symptoms.strip(),
                baseline_url=baseline_url,
                qwen_url=qwen_url,
                mistral_url=mistral_url,
                mistral_api_key=mistral_api_key,
                age=age,
                gender=gender,
            )
        except ModelRequestError as error:
            st.error(str(error))
            st.stop()

    st.success(f"Response received from {model_label}.")

    confidence = result.get("confidence")
    if confidence is not None:
        st.caption(f"Model confidence: {confidence}%")

    training_metrics = result.get("training_metrics") or {}
    if training_metrics:
        st.caption(
            "Checkpoint held-out chunk metrics — "
            f"Top-1: {training_metrics.get('held_out_chunk_top_1', 0):.1%}, "
            f"Top-3: {training_metrics.get('held_out_chunk_top_3', 0):.1%}"
        )
    if result.get("selection_mode") == "rag_quality_guard":
        st.info(
            "The checkpoint is still below the project quality guard, so the displayed condition uses the "
            "top local RAG match. The deep-model ranking remains available for training evaluation."
        )

    patient_context = result.get("patient_context") or {}
    if patient_context.get("age") is not None or patient_context.get("gender"):
        st.caption("Optional age/gender context was included in this request.")

    left, right = st.columns(2)
    with left:
        show_result_card("Condition", result["condition"])
        show_result_card("Symptoms", result["symptoms"])
        show_result_card("Causes", result["causes"])
    with right:
        show_result_card("Warnings", result["warnings"])
        show_result_card("Recommendations", result["recommendations"])

    with st.expander("Technical response"):
        st.json(result)
