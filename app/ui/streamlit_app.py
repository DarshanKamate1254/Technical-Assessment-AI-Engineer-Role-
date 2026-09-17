"""Streamlit UI for Customer Support AI Assistant.

This frontend communicates EXCLUSIVELY with the FastAPI backend via HTTP (APIClient).
It contains NO direct imports of SQLite, analytics, anomalies, LangChain, or Groq.
"""

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
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-tool {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-family: monospace;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #BFDBFE;
    }
    .badge-severity-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-severity-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-severity-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .evidence-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 12px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: System Status & Quick Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/service.png", width=64)
    st.title("Support AI Hub")
    st.caption("AI-Powered Support Analytics & Incident Insights")

    st.divider()

    # Backend Connection Status Check
    is_healthy = client.get_health()
    if is_healthy:
        st.success(f"🟢 Backend Connected\n`{API_BASE_URL}`")
    else:
        st.error(
            f"🔴 Backend Disconnected\n`{API_BASE_URL}`\n\n"
            "Unable to connect to the support analytics API. "
            "Please make sure the FastAPI server is running."
        )

    st.divider()
    st.markdown("### 💡 Example Questions")
    example_prompts = [
        "How many critical tickets are unresolved?",
        "Which agent has the lowest average customer rating?",
        "How many unresolved tickets are older than 24 hours?",
        "Show high-priority unresolved tickets older than 24 hours.",
        "What is the average resolution time by priority?",
        "Are there any anomalies in resolution times?",
    ]

    selected_example = None
    for i, prompt in enumerate(example_prompts):
        if st.button(f"📌 {prompt}", key=f"ex_btn_{i}", use_container_width=True):
            st.session_state["active_question"] = prompt

    st.divider()
    st.caption("Stage 7 • FastAPI + Streamlit Interface")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Customer Support AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Natural Language Ticket Intelligence & Operational Insights</div>',
    unsafe_allow_html=True,
)

# Tab Navigation
tab_qa, tab_dashboard, tab_anomalies = st.tabs(
    ["💬 AI Assistant", "📊 Operational Dashboard", "🚨 Anomaly Detection"]
)

# =============================================================================
# TAB 1: AI Assistant (POST /ask)
# =============================================================================
with tab_qa:
    st.markdown("### Ask a Question About Support Tickets")

    # Question Input
    default_text = st.session_state.get("active_question", "")
    user_query = st.text_input(
        "Ask a question about support tickets...",
        value=default_text,
        placeholder="e.g., How many critical tickets are unresolved?",
        key="query_input",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        submit_clicked = st.button("🚀 Ask", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear", use_container_width=False):
            st.session_state["active_question"] = ""
            st.rerun()

    if submit_clicked and user_query:
        if not is_healthy:
            st.error(
                "Unable to connect to the support analytics API. "
                "Please make sure the FastAPI server is running."
            )
        else:
            with st.spinner("Analyzing support database and computing deterministic metrics..."):
                response = client.ask_question(user_query)

            if not response.get("success", True) and response.get("error"):
                st.warning(f"⚠️ {response.get('answer')}")
            else:
                st.markdown("#### 💡 Answer")
                st.info(response.get("answer", "No answer provided."))

                # Tool Information Display
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    st.markdown("**Tools Executed:**")
                    tool_html = " ".join(
                        [
                            f'<span class="badge-tool">⚙️ {tc.get("tool")}</span>'
                            for tc in tool_calls
                        ]
                    )
                    st.markdown(tool_html, unsafe_allow_html=True)

                # Evidence Display
                evidence = response.get("evidence", {})
                if evidence:
                    with st.expander("🔍 Provenance & Structured Evidence", expanded=True):
                        st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
                        st.json(evidence)
                        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2: Operational Dashboard (GET /analytics/summary)
# =============================================================================
with tab_dashboard:
    st.markdown("### Operational Ticket Analytics")

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
            # Metric KPIs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tickets", f"{summary.get('total_tickets', 0):,}")
            col2.metric("Resolved", f"{summary.get('resolved_tickets', 0):,}")
            col3.metric("Unresolved", f"{summary.get('unresolved_tickets', 0):,}")
            col4.metric(
                "Critical Unresolved",
                f"{summary.get('critical_unresolved', 0):,}",
                delta_color="inverse",
            )

            st.divider()

            # Breakdown Visualizations
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### Backlog by Priority")
                df_priority = pd.DataFrame(summary.get("by_priority", []))
                if not df_priority.empty:
                    st.dataframe(df_priority, use_container_width=True, hide_index=True)

                st.markdown("#### Backlog by Category")
                df_category = pd.DataFrame(summary.get("by_category", []))
                if not df_category.empty:
                    st.dataframe(df_category, use_container_width=True, hide_index=True)

            with col_right:
                st.markdown("#### Status Breakdown")
                df_status = pd.DataFrame(summary.get("by_status", []))
                if not df_status.empty:
                    st.dataframe(df_status, use_container_width=True, hide_index=True)

                st.markdown("#### Performance Metrics")
                resp_info = summary.get("response_time", {})
                res_info = summary.get("resolution_time", {})
                rating_info = summary.get("customer_rating", {})

                perf_data = {
                    "Metric": [
                        "Avg Response Time (hrs)",
                        "Avg Resolution Time (hrs)",
                        "Avg Customer Rating",
                    ],
                    "Value": [
                        f"{resp_info.get('average_response_time_hrs', 0):.2f} hrs",
                        f"{res_info.get('average_resolution_time_hrs', 0):.2f} hrs",
                        f"⭐ {rating_info.get('average_customer_rating', 0):.2f} / 5.0",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(perf_data), use_container_width=True, hide_index=True
                )

# =============================================================================
# TAB 3: Anomaly Detection (GET /anomalies)
# =============================================================================
with tab_anomalies:
    st.markdown("### Anomaly Detection & SLA Outliers")

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

            # Severity Counters
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Anomalies", summary_stats.get("total_anomalies", 0))
            c2.metric("Critical", summary_stats.get("critical", 0))
            c3.metric("High", summary_stats.get("high", 0))
            c4.metric("Medium", summary_stats.get("medium", 0))

            st.divider()

            # Statistical Thresholds Info
            stat_thresholds = anomalies_data.get("statistical_thresholds", {})
            if stat_thresholds:
                st.caption(
                    f"📐 Statistical Bounds (IQR 1.5x) — "
                    f"Resolution Time Upper Bound: **{stat_thresholds.get('resolution_time_upper_bound_hrs', 0)} hrs** "
                    f"(Outliers: {stat_thresholds.get('resolution_outlier_count', 0)}) | "
                    f"Response Time Upper Bound: **{stat_thresholds.get('response_time_upper_bound_hrs', 0)} hrs** "
                    f"(Outliers: {stat_thresholds.get('response_outlier_count', 0)})"
                )

            # Anomaly Records Table
            if anomalies_list:
                df_anomalies = pd.DataFrame(anomalies_list)

                # Severity filter
                severities = ["All"] + sorted(list(df_anomalies["severity"].unique()))
                selected_sev = st.selectbox("Filter by Severity:", severities)

                if selected_sev != "All":
                    display_df = df_anomalies[df_anomalies["severity"] == selected_sev]
                else:
                    display_df = df_anomalies

                cols_to_show = [
                    col
                    for col in [
                        "ticket_id",
                        "anomaly_type",
                        "severity",
                        "priority",
                        "status",
                        "agent_id",
                        "reason",
                    ]
                    if col in display_df.columns
                ]

                st.dataframe(
                    display_df[cols_to_show],
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("🔍 View Raw Anomaly Payload"):
                    st.json(anomalies_list)
            else:
                st.success("No operational anomalies detected in the current ticket dataset.")
