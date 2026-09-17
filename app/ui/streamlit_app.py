"""Streamlit UI for Customer Support AI Assistant.

This frontend communicates EXCLUSIVELY with the FastAPI backend via HTTP (APIClient).
It contains NO direct imports of SQLite, analytics, anomalies, LangChain, or Groq.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path when running directly with streamlit
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from app.ui.api_client import APIClient
except ModuleNotFoundError:
    from api_client import APIClient  # type: ignore

# -----------------------------------------------------------------------------
# Configuration & Client Setup
# -----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Support AI Assistant",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize API Client
client = APIClient(base_url=API_BASE_URL)

# -----------------------------------------------------------------------------
# Premium Custom Styling & CSS Design System
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
    }

    /* Answer Container Card */
    .answer-card {
        background: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        position: relative;
        transition: all 0.2s ease-in-out;
    }
    .answer-card:hover {
        box-shadow: 0 10px 30px -4px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .answer-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .answer-badge {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .answer-body {
        font-size: 1.05rem;
        line-height: 1.65;
        color: #1E293B;
    }

    /* Tool Call Badges */
    .tool-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #475569;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 8px;
        margin-right: 8px;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .tool-chip-accent {
        color: #2563EB;
        border-color: #BFDBFE;
        background: #EFF6FF;
    }

    /* Example prompt chips */
    .prompt-chip {
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #334155;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    /* Severity Indicators */
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    /* Evidence Box */
    .evidence-container {
        background: #F8FAFC;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        padding: 16px;
        margin-top: 14px;
    }

    /* Metric card adjustments */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: System Status & Backend Info (Cleaned - No Example Prompts)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/service.png", width=56)
    st.markdown("### **Support AI Hub**")
    st.caption("AI-Powered Support Analytics & Incident Insights")

    st.markdown("---")

    # Backend Connectivity
    is_healthy = client.get_health()
    if is_healthy:
        st.success("🟢 **Backend Connected**")
        st.caption(f"Endpoint: `{API_BASE_URL}`")
    else:
        st.error("🔴 **Backend Offline**")
        st.caption(f"Target: `{API_BASE_URL}`")
        st.warning(
            "FastAPI server is unreachable.\n"
            "Run: `uvicorn app.api.main:app --reload`"
        )

    st.markdown("---")
    st.markdown("**System Architecture**")
    st.caption("• **Frontend**: Streamlit HTTP Client\n• **API**: FastAPI (port 8000)\n• **Orchestrator**: LangChain Agent\n• **Inference**: Groq LLM\n• **Database**: SQLite (500 tickets)")

    st.markdown("---")
    st.caption("Stage 7 • Technical Assessment AI System")


# -----------------------------------------------------------------------------
# Main Banner Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Customer Support AI Assistant</div>
        <div class="hero-subtitle">Natural language support ticket analytics, SLA tracking & anomaly detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Session States
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

# Tab Navigation
tab_qa, tab_dashboard, tab_anomalies = st.tabs(
    ["💬 AI Assistant", "📊 Operational Dashboard", "🚨 Anomaly Detection"]
)

# =============================================================================
# TAB 1: AI Assistant (POST /ask) with Enter-to-Submit Form
# =============================================================================
with tab_qa:
    st.markdown("#### 💡 Suggested Inquiries")

    # Suggestion Chips in Main Page (Click to populate query)
    example_prompts = [
        "How many critical tickets are unresolved?",
        "Which agent has the lowest average customer rating?",
        "How many unresolved tickets are older than 24 hours?",
        "Show high-priority unresolved tickets older than 24 hours.",
        "What is the average resolution time by priority?",
        "Are there any anomalies in resolution times?",
    ]

    cols = st.columns(3)
    for idx, prompt in enumerate(example_prompts):
        col_target = cols[idx % 3]
        with col_target:
            if st.button(f"👉 {prompt}", key=f"quick_btn_{idx}", use_container_width=True):
                st.session_state["prompt_to_run"] = prompt

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Check if a quick button was clicked
    preset_query = st.session_state.pop("prompt_to_run", "")

    # Natural Language Query Form (Allows ENTER key to execute immediately)
    with st.form(key="ask_form", clear_on_submit=False):
        user_query = st.text_input(
            "Ask a question about support tickets...",
            value=preset_query,
            placeholder="Type your question and press Enter or click Ask...",
            help="Press Enter to execute the question immediately.",
            key="user_query_input",
        )
        col_submit, col_hint = st.columns([1, 4])
        with col_submit:
            submit_button = st.form_submit_button("🚀 Ask", type="primary", use_container_width=True)
        with col_hint:
            st.caption("⌨️ *Tip: You can hit **Enter** directly in the text box to submit!*")

    # Trigger Execution on Submit
    if (submit_button or preset_query) and (user_query or preset_query):
        active_q = user_query.strip() or preset_query.strip()
        if not is_healthy:
            st.error(
                "Unable to connect to the support analytics API. "
                "Please make sure the FastAPI server is running."
            )
        elif active_q:
            with st.spinner("🤖 Orchestrating deterministic analytics & querying data store..."):
                response_data = client.ask_question(active_q)
                st.session_state["last_query"] = active_q
                st.session_state["last_response"] = response_data

    # Display Answer and Evidence
    if st.session_state.get("last_response"):
        resp = st.session_state["last_response"]
        q_text = st.session_state.get("last_query", "")

        if not resp.get("success", True) and resp.get("error"):
            st.warning(f"⚠️ {resp.get('answer', 'An error occurred.')}")
        else:
            # Styled Beautiful Answer Card
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-header">
                        <span class="answer-badge">✨ AI Response</span>
                        <span style="color: #64748B; font-size: 0.9rem;">Query: <b>{q_text}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Answer content rendered with full Markdown support
            st.markdown(resp.get("answer", "No answer generated."))

            # Tool Provenance
            tool_calls = resp.get("tool_calls", [])
            if tool_calls:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.markdown("**Deterministic Tools Invoked:**")
                tool_chips = "".join(
                    [
                        f'<span class="tool-chip tool-chip-accent">⚙️ {tc.get("tool")}</span>'
                        for tc in tool_calls
                    ]
                )
                st.markdown(tool_chips, unsafe_allow_html=True)

            # Provenance & Structured Evidence Expander
            evidence = resp.get("evidence", {})
            if evidence:
                with st.expander("🔍 Provenance & Structured Evidence Payload", expanded=False):
                    st.json(evidence)


# =============================================================================
# TAB 2: Operational Dashboard (GET /analytics/summary)
# =============================================================================
with tab_dashboard:
    st.markdown("### 📊 Operational Ticket Analytics")
    st.caption("Real-time summary aggregated deterministically from the SQLite ticket repository.")

    if not is_healthy:
        st.error(
            "Unable to connect to the support analytics API. "
            "Please make sure the FastAPI server is running."
        )
    else:
        summary = client.get_summary()
        if not summary:
            st.error("Failed to retrieve analytics summary from the API.")
        else:
            # Executive Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Tickets", f"{summary.get('total_tickets', 0):,}")
            m2.metric("Resolved Tickets", f"{summary.get('resolved_tickets', 0):,}")
            m3.metric("Unresolved Backlog", f"{summary.get('unresolved_tickets', 0):,}")
            m4.metric(
                "Critical Unresolved",
                f"{summary.get('critical_unresolved', 0):,}",
                delta=f"{summary.get('critical_unresolved', 0)} requiring attention",
                delta_color="inverse",
            )

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # Categorical Breakdowns
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("#### Backlog by Priority")
                df_priority = pd.DataFrame(summary.get("by_priority", []))
                if not df_priority.empty:
                    st.dataframe(df_priority, use_container_width=True, hide_index=True)

                st.markdown("#### Backlog by Category")
                df_category = pd.DataFrame(summary.get("by_category", []))
                if not df_category.empty:
                    st.dataframe(df_category, use_container_width=True, hide_index=True)

            with col_r:
                st.markdown("#### Status Distribution")
                df_status = pd.DataFrame(summary.get("by_status", []))
                if not df_status.empty:
                    st.dataframe(df_status, use_container_width=True, hide_index=True)

                st.markdown("#### Key Performance Indicators")
                resp_info = summary.get("response_time", {})
                res_info = summary.get("resolution_time", {})
                rating_info = summary.get("customer_rating", {})

                perf_df = pd.DataFrame(
                    {
                        "KPI Metric": [
                            "Average Response Time",
                            "Average Resolution Time",
                            "Average Customer Satisfaction",
                        ],
                        "Observed Value": [
                            f"{resp_info.get('average_response_time_hrs', 0):.2f} hours",
                            f"{res_info.get('average_resolution_time_hrs', 0):.2f} hours",
                            f"⭐ {rating_info.get('average_customer_rating', 0):.2f} / 5.0",
                        ],
                    }
                )
                st.dataframe(perf_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3: Anomaly Detection (GET /anomalies)
# =============================================================================
with tab_anomalies:
    st.markdown("### 🚨 Anomaly Detection & SLA Breaches")
    st.caption("Deterministic business-rule violations and statistical IQR resolution-time outlier reports.")

    if not is_healthy:
        st.error(
            "Unable to connect to the support analytics API. "
            "Please make sure the FastAPI server is running."
        )
    else:
        anomalies_data = client.get_anomalies()
        if not anomalies_data:
            st.error("Failed to retrieve anomaly reports from the API.")
        else:
            summary_stats = anomalies_data.get("summary", {})
            anomalies_list = anomalies_data.get("anomalies", [])

            # Anomaly KPIs
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Total Flagged", summary_stats.get("total_anomalies", 0))
            a2.metric("Critical Severity", summary_stats.get("critical", 0))
            a3.metric("High Severity", summary_stats.get("high", 0))
            a4.metric("Medium Severity", summary_stats.get("medium", 0))

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Statistical IQR Context
            stat_thresholds = anomalies_data.get("statistical_thresholds", {})
            if stat_thresholds:
                st.info(
                    f"📐 **Statistical Baseline (IQR 1.5×)** — "
                    f"Resolution Time Upper Bound: **{stat_thresholds.get('resolution_time_upper_bound_hrs', 0)} hrs** "
                    f"({stat_thresholds.get('resolution_outlier_count', 0)} outliers) | "
                    f"Response Time Upper Bound: **{stat_thresholds.get('response_time_upper_bound_hrs', 0)} hrs** "
                    f"({stat_thresholds.get('response_outlier_count', 0)} outliers)"
                )

            # Anomalies Table
            if anomalies_list:
                df_anomalies = pd.DataFrame(anomalies_list)

                # Severity filter
                sev_options = ["All"] + sorted(list(df_anomalies["severity"].unique()))
                selected_sev = st.selectbox("Filter Anomaly Table by Severity:", sev_options)

                if selected_sev != "All":
                    filtered_df = df_anomalies[df_anomalies["severity"] == selected_sev]
                else:
                    filtered_df = df_anomalies

                cols_display = [
                    c
                    for c in [
                        "ticket_id",
                        "anomaly_type",
                        "severity",
                        "priority",
                        "status",
                        "agent_id",
                        "reason",
                    ]
                    if c in filtered_df.columns
                ]

                st.dataframe(
                    filtered_df[cols_display],
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("🔍 View Complete Raw Anomaly JSON"):
                    st.json(anomalies_list)
            else:
                st.success("No operational anomalies detected in the support ticket database.")
