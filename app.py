"""
SPY Pattern Analysis Dashboard - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Import our organized utility modules
from utils import (
    load_data, 
    generate_cache, 
    analyze_current_day_pattern_outcomes,
    build_projection_chart,
    compute_stepwise_highlights,
    render_highlights,
    cache_exists_in_s3,
    load_cache_from_s3,
    fetch_cycle_pattern_data,
    fetch_secondary_patterns
)
from config import DATA_DIR, CSV_FILENAME, DEFAULT_PROJ_DAYS, DEFAULT_REAL_BARS, DEFAULT_WINDOW_SIZE

# Page configuration
st.set_page_config(page_title="SPY Pattern Mapper — Per-Pattern Projections", layout="wide")

# Load data
@st.cache_data
def load_stock_data():
    """Load stock data with caching"""
    data_path = Path(__file__).resolve().parent / DATA_DIR / CSV_FILENAME
    return load_data(data_path)

df = load_stock_data()
df["Index"] = df.index

# Projections state
if "show_proj" not in st.session_state:
    st.session_state["show_proj"] = False

# Main title
st.title("📈 SPY — Pattern Analysis Dashboard")

# ==== TOP: Pure Candlestick Chart (no projections) ====
st.subheader("Candlestick Chart")

# Bar count filter (shared across click projections and manual analysis)
st.subheader("🔍 Pattern Filter")
bar_count = st.selectbox(
    "Filter by number of bars:",
    options=["All", "2", "3", "4", "5", "6", "7"],
    index=0,
    help="Select the number of bars to filter pattern mappers. For example, '2' will show patterns like 4324/4434, '3' will show 4434/4134/3442, etc."
)
st.session_state["bar_count"] = bar_count

# Limit number of overlay patterns per date for performance
max_overlay = st.slider(
    "Max overlay patterns per date",
    min_value=1,
    max_value=10,
    value=3,
    help="Limits how many pattern projections are drawn per clicked date on the main chart."
)

# Window size control
window_size = st.number_input("Days to show:", min_value=5, max_value=200, value=DEFAULT_WINDOW_SIZE, step=5)

# Projection controls (button-triggered; no click listeners)
proj_col1, proj_col2 = st.columns([3, 2])
with proj_col1:
    if st.button("Show projections", type="primary"):
        st.session_state["show_proj"] = True
with proj_col2:
    recent_bars = st.slider(
        "Bars to project from",
        min_value=1,
        max_value=10,
        value=6,
        step=1,
        help="Number of most recent trading days used as starting points for projections.",
    )

# Build full-history trace, then set initial x-range & y-range to the window
fig_top = go.Figure(
    data=[
        go.Candlestick(
            x=df["Index"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="SPY",
            hoverinfo="text",
            hovertext=[
                f"Date: {date}<br>Open: {op}<br>High: {hi}<br>Low: {lo}<br>Close: {cl}" 
                for date, op, hi, lo, cl in zip(
                    df["Date"].dt.strftime("%Y-%m-%d"),
                    df["Open"],
                    df["High"],
                    df["Low"],
                    df["Close"]
                )
            ],
        )
    ]
)

# Initial x-range (last N days)
max_index = len(df) - 1
start_idx = max(0, max_index - int(window_size) + 1)
end_idx = max_index

fig_top.update_layout(
    template="plotly_dark",
    xaxis_title="Trading Days",
    yaxis_title="Price (USD)",
    height=600,
    xaxis_rangeslider_visible=False,
    showlegend=False,
    dragmode="pan",
    uirevision="keep",
)

# Set initial x-range by index (continuous, no gaps)
fig_top.update_xaxes(range=[start_idx, end_idx])

# Auto-scale Y to the visible window only
window_slice = df.iloc[start_idx:end_idx + 1]
w_low = float(window_slice["Low"].min())
w_high = float(window_slice["High"].max())
span = max(1e-6, w_high - w_low)
pad = 0.05 * span
fig_top.update_yaxes(range=[w_low - pad, w_high + pad])

# Map index ticks back to dates for readability
tick_step = max(1, (end_idx - start_idx) // 10)
tick_vals = list(range(start_idx, end_idx + 1, tick_step))
tick_text = [df["Date"].iloc[i].strftime("%Y-%m-%d") for i in tick_vals]
fig_top.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=45,
    )
)

# Cached helpers to speed up overlay computations
@st.cache_data(show_spinner=False)
def get_patterns_data_cached(sel_date: str):
    filename = f"{sel_date}_patterns_cache.json"
    if not cache_exists_in_s3(filename):
        gen_name = generate_cache(df, current_date=sel_date)
        if gen_name is None:
            return None
        filename = gen_name
    return load_cache_from_s3(filename)

@st.cache_data(show_spinner=False)
def compute_outcomes_cached(occurrences: tuple[str, ...], sel_date: str, look_ahead: int):
    return analyze_current_day_pattern_outcomes(df, list(occurrences), sel_date, look_ahead=look_ahead)

# Button-driven overlay projections for the most recent 6 trading days
if st.session_state.get("show_proj"):
    overlay_min = None
    overlay_max = None
    overlay_end_idx = end_idx

    # Last N trading dates (user-selected) as strings
    n_recent = int(recent_bars)
    sel_dates = df["Date"].iloc[max(0, len(df) - n_recent):].dt.strftime("%Y-%m-%d").tolist()

    colors = [
        "#FF6B6B", "#4ECDC4", "#FFD93D", "#1A73E8", "#C084FC", "#00C853", "#FF8A65", "#F4511E"
    ]
    dtw = 0.3
    for i, sel_date in enumerate(sel_dates):
        color = colors[i % len(colors)]
        # Locate selected date index and close
        try:
            sel_dt = pd.to_datetime(sel_date).normalize()
            hits = df.index[df["Date"].dt.normalize() == sel_dt].tolist()
            if not hits:
                continue
            pattern_idx = hits[0]
            pattern_close = float(df["Close"].iloc[pattern_idx])
        except Exception:
            continue

        # Mark the selected date on the chart
        fig_top.add_vline(x=pattern_idx, line_width=1, line_dash="dot", line_color=color)

        # Load or generate cache for this date (cached)
        patterns_data = get_patterns_data_cached(sel_date)
        if not patterns_data:
            continue

        # Select matching patterns and limit count for performance
        matching = []
        for key, val in patterns_data.items():
            if val.get("occurrences_count", 0) <= 0:
                continue
            if "_steps_" not in key:
                continue
            try:
                steps = int(key.split("_steps_")[1])
            except Exception:
                continue
            if bar_count != "All" and str(steps) != bar_count:
                continue
            matching.append((key, val))

        # Sort by occurrences_count desc and take top N
        matching.sort(key=lambda kv: kv[1].get("occurrences_count", 0), reverse=True)
        matching = matching[:max_overlay]

        for key, val in matching:
            # Compute outcomes for this pattern's matches (cached)
            try:
                occ_tuple = tuple(val.get("occurrences", []))
                df_outcomes = compute_outcomes_cached(occ_tuple, sel_date, DEFAULT_PROJ_DAYS)
            except Exception:
                continue

            # Draw dominance-based horizontal segments for k=1..lookahead
            for k in range(1, min(DEFAULT_PROJ_DAYS, len(df_outcomes)) + 1):
                row = df_outcomes.loc[str(k)] if str(k) in df_outcomes.index else None
                if row is None:
                    continue
                dom = float(row.get("dominant_probability", 0.0))
                pos_rng = float(row.get("positive_range", 0.0) or 0.0)
                neg_rng = float(row.get("negative_range", 0.0) or 0.0)

                # Determine color based on dominance
                if dom > 0:
                    y_lvl = pattern_close + pos_rng
                    proj_color = "blue"
                elif dom < 0:
                    y_lvl = pattern_close + neg_rng
                    proj_color = "red"
                else:
                    y_lvl = pattern_close + (pos_rng + neg_rng) / 2
                    proj_color = "grey"

                proj_idx = pattern_idx + k
                fig_top.add_trace(go.Scatter(
                    x=[proj_idx - dtw, proj_idx + dtw],
                    y=[y_lvl, y_lvl],
                    mode="lines",
                    line=dict(dash="dash", width=2, color=proj_color),
                    showlegend=False,
                    hovertemplate=f"<b>{sel_date}</b><br>P+{k}: %{{y:.2f}}<br>Dom: {dom:.2f}<extra></extra>",
                ))

                # Track overlay min/max and the furthest projected index
                overlay_min = y_lvl if overlay_min is None else min(overlay_min, y_lvl)
                overlay_max = y_lvl if overlay_max is None else max(overlay_max, y_lvl)
                overlay_end_idx = max(overlay_end_idx, proj_idx)

    # If overlays extend beyond current Y range, expand Y to include them
    if overlay_min is not None and overlay_max is not None:
        base_low = w_low
        base_high = w_high
        new_low = min(base_low, overlay_min)
        new_high = max(base_high, overlay_max)
        span2 = max(1e-6, new_high - new_low)
        pad2 = 0.05 * span2
        fig_top.update_yaxes(range=[new_low - pad2, new_high + pad2])

    # If overlays extend beyond current x-range, extend to show projections
    if overlay_end_idx > end_idx:
        fig_top.update_xaxes(range=[start_idx, overlay_end_idx])

# Show the main chart (static, no click listeners)
st.plotly_chart(fig_top, use_container_width=True, key="main_candles_chart")

# (Removed click-driven projections section)

# ==== Pattern Analysis ====
st.header("🧩 Pattern Analysis")
last_date = df["Date"].iloc[-1]
pattern_date = st.date_input(
    "Select date for pattern analysis:",
    value=last_date
).strftime("%Y-%m-%d")

bar_count = st.session_state.get("bar_count", "All")

# Add analyze button
if st.button("📊 Analyze Patterns", type="primary"):
    # Check if cache exists and generate if needed
    filename = f"{pattern_date}_patterns_cache.json"

    if not cache_exists_in_s3(filename):
        with st.spinner(f"Generating pattern cache for {pattern_date}..."):
            filename = generate_cache(df, current_date=pattern_date)
            if filename is None:
                st.error(f"❌ The selected date ({pattern_date}) was not a trading day.")
                st.stop()
            else:
                st.success(f"✅ Cache generated in S3: {filename}")
    else:
        st.info(f"Using existing cache file from S3: {filename}")

    patterns_data = load_cache_from_s3(filename)

    # Display pattern analysis with filtering
    filtered_patterns = 0
    total_patterns = 0
    
    for key, val in patterns_data.items():
        if val.get("occurrences_count", 0) <= 0:
            continue
            
        total_patterns += 1
        
        # Extract steps from key (format: "gap_X_steps_Y")
        if "_steps_" in key:
            steps = int(key.split("_steps_")[1])
        else:
            continue
            
        # Apply bar count filter
        if bar_count != "All" and str(steps) != bar_count:
            continue
            
        filtered_patterns += 1
        
        df_outcomes = analyze_current_day_pattern_outcomes(
            df, val["occurrences"], pattern_date, look_ahead=DEFAULT_PROJ_DAYS
        )

        with st.expander(f"{key} → {val['target_pattern']}  (occurrences: {val['occurrences_count']})", expanded=False):
            # Highlights row
            highlights = compute_stepwise_highlights(df_outcomes)
            render_highlights(highlights)
            st.dataframe(df_outcomes.T)

            fig_proj = build_projection_chart(
                df_full=df,
                outcomes_df=df_outcomes,
                pattern_date=pattern_date,
                look_ahead=DEFAULT_PROJ_DAYS,
                real_candles=DEFAULT_REAL_BARS,
                highlights=highlights
            )
            st.plotly_chart(fig_proj, use_container_width=True, key=f"proj_chart_{key}")
    
    # Show filtering results
    if bar_count != "All":
        st.info(f"🔍 Showing {filtered_patterns} patterns with {bar_count} bars (out of {total_patterns} total patterns)")
    else:
        st.info(f"📊 Showing all {total_patterns} patterns")
else:
    st.info("👆 Click 'Analyze Patterns' button to start pattern analysis")

