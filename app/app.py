from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Rail Operations Forecaster",
    page_icon="🚆",
    layout="wide"
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

@st.cache_data
def load_dashboard_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "dashboard_forecasts.csv"

    if not data_path.exists():
        st.error(
            "Dashboard data file not found. Please run "
            "`notebooks/12_dashboard_data_prep.ipynb` first to generate "
            "`data/processed/dashboard_forecasts.csv`."
        )
        st.stop()

    df = pd.read_csv(data_path, parse_dates=["date"])
    return df


df = load_dashboard_data()


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def format_pct(value):
    return f"{value:.1%}"


def risk_tier_order():
    return [
        "Normal",
        "Elevated Risk",
        "High Risk",
        "Threshold Breach Warning",
    ]


def priority_order():
    return [
        "Normal",
        "Monitor",
        "Planning Review",
        "Watchlist",
        "Urgent Review",
    ]


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("🚆 Rail Operations Forecaster")
st.caption(
    "Interactive decision-support dashboard for terminal dwell forecasting, "
    "breach early-warning, and operational risk monitoring."
)

st.markdown(
    """
    This dashboard summarizes outputs from the Rail Operations Forecaster project.
    It uses a synthetic rail-style terminal dataset and combines:
    
    - tuned LightGBM dwell forecasting
    - operational risk tiers
    - dedicated 24-hour breach classification
    - dashboard priority labels
    - plain-English operating summaries
    """
)


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------

st.sidebar.header("Dashboard Filters")

terminal_options = sorted(df["terminal_id"].unique())
selected_terminals = st.sidebar.multiselect(
    "Select terminals",
    options=terminal_options,
    default=terminal_options,
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

risk_options = risk_tier_order()
available_risk_options = [risk for risk in risk_options if risk in df["risk_tier"].unique()]

selected_risk_tiers = st.sidebar.multiselect(
    "Select risk tiers",
    options=available_risk_options,
    default=available_risk_options,
)

priority_options = priority_order()
available_priority_options = [
    priority for priority in priority_options
    if priority in df["dashboard_priority"].unique()
]

selected_priorities = st.sidebar.multiselect(
    "Select dashboard priorities",
    options=available_priority_options,
    default=available_priority_options,
)


# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------

filtered_df = df.copy()

if selected_terminals:
    filtered_df = filtered_df[filtered_df["terminal_id"].isin(selected_terminals)]

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date) &
        (filtered_df["date"].dt.date <= end_date)
    ]

if selected_risk_tiers:
    filtered_df = filtered_df[filtered_df["risk_tier"].isin(selected_risk_tiers)]

if selected_priorities:
    filtered_df = filtered_df[filtered_df["dashboard_priority"].isin(selected_priorities)]


# ------------------------------------------------------------
# Executive summary cards
# ------------------------------------------------------------

st.subheader("Executive Summary")

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

avg_predicted_dwell = filtered_df["predicted_dwell"].mean()
avg_actual_dwell = filtered_df["target_dwell_hours"].mean()
classifier_warning_count = int(filtered_df["classifier_warning_flag"].sum())
actual_breach_count = int(filtered_df["actual_breach_24h"].sum())
high_risk_count = int(
    filtered_df["risk_tier"].isin(["High Risk", "Threshold Breach Warning"]).sum()
)

highest_risk_row = filtered_df.sort_values(
    ["classifier_breach_probability", "predicted_dwell"],
    ascending=False
).iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Avg Predicted Dwell", f"{avg_predicted_dwell:.1f}h")
col2.metric("Avg Actual Dwell", f"{avg_actual_dwell:.1f}h")
col3.metric("High-Risk Rows", f"{high_risk_count:,}")
col4.metric("Classifier Warnings", f"{classifier_warning_count:,}")
col5.metric("Actual Breaches", f"{actual_breach_count:,}")

st.info(
    f"Highest-risk filtered row: Terminal **{highest_risk_row['terminal_id']}** on "
    f"**{highest_risk_row['date'].date()}** with classifier breach probability "
    f"**{highest_risk_row['classifier_breach_probability']:.1%}** and predicted dwell "
    f"**{highest_risk_row['predicted_dwell']:.1f} hours**."
)


# ------------------------------------------------------------
# Risk tier and priority distribution
# ------------------------------------------------------------

st.subheader("Risk and Priority Distribution")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    risk_counts = (
        filtered_df["risk_tier"]
        .value_counts()
        .reindex(risk_tier_order())
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    risk_counts.plot(kind="bar", ax=ax)
    ax.set_title("Risk Tier Counts")
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Rows")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with dist_col2:
    priority_counts = (
        filtered_df["dashboard_priority"]
        .value_counts()
        .reindex(priority_order())
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    priority_counts.plot(kind="bar", ax=ax)
    ax.set_title("Dashboard Priority Counts")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Rows")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


# ------------------------------------------------------------
# Terminal trend view
# ------------------------------------------------------------

st.subheader("Terminal Trend View")

selected_terminal_for_trend = st.selectbox(
    "Select one terminal for trend view",
    options=sorted(filtered_df["terminal_id"].unique())
)

trend_df = (
    filtered_df[filtered_df["terminal_id"] == selected_terminal_for_trend]
    .sort_values("date")
    .copy()
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(trend_df["date"], trend_df["target_dwell_hours"], label="Actual dwell")
ax.plot(trend_df["date"], trend_df["predicted_dwell"], label="Predicted dwell")
ax.axhline(24, linestyle="--", label="24-hour breach threshold")
ax.set_title(f"Actual vs Predicted Dwell — Terminal {selected_terminal_for_trend}")
ax.set_xlabel("Date")
ax.set_ylabel("Dwell hours")
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig)


# ------------------------------------------------------------
# High-risk table
# ------------------------------------------------------------

st.subheader("Terminal Risk Table")

display_cols = [
    "date",
    "terminal_id",
    "terminal_name",
    "region",
    "target_dwell_hours",
    "predicted_dwell",
    "risk_tier",
    "classifier_breach_probability",
    "classifier_warning_flag",
    "resource_pressure_flag",
    "dashboard_priority",
    "yard_occupancy_pct",
    "crew_starts_available",
    "locomotive_availability_pct",
]

table_df = filtered_df[display_cols].copy()
table_df = table_df.sort_values(
    ["classifier_breach_probability", "predicted_dwell"],
    ascending=False
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)


# ------------------------------------------------------------
# Plain-English operating summaries
# ------------------------------------------------------------

st.subheader("Plain-English Operating Summaries")

summary_count = st.slider(
    "Number of summaries to show",
    min_value=3,
    max_value=15,
    value=5,
    step=1,
)

summary_df = table_df.head(summary_count).merge(
    filtered_df[["date", "terminal_id", "plain_english_summary"]],
    on=["date", "terminal_id"],
    how="left"
)

for _, row in summary_df.iterrows():
    st.markdown(
        f"""
        **Terminal {row['terminal_id']} — {row['date'].date()}**  
        Priority: **{row['dashboard_priority']}**  
        Risk tier: **{row['risk_tier']}**  
        Classifier breach probability: **{row['classifier_breach_probability']:.1%}**  
        
        {row['plain_english_summary']}
        """
    )
    st.divider()


# ------------------------------------------------------------
# Project notes
# ------------------------------------------------------------

with st.expander("Project interpretation notes"):
    st.markdown(
        """
        - The regression model supports dwell magnitude forecasting and planning.
        - The classifier supports 24-hour breach early-warning.
        - Risk tiers translate predicted dwell into operational categories.
        - Dashboard priorities combine risk-tier and classifier-warning logic.
        - This dashboard uses synthetic data and should be interpreted as a portfolio-safe decision-support demonstration.
        - It does not represent actual BNSF or proprietary railroad operations.
        """
    )