"""
SPY Pattern Analysis Dashboard - Main Streamlit Application
"""


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from statistics import mean  # for aggregated lines

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
    help="Select the number of bars to filter pattern mappers.",
)
st.session_state["bar_count"] = bar_count

# Limit number of overlay patterns per date for performance
max_overlay = st.slider(
    "Max overlay patterns per date",
    min_value=1,
    max_value=10,
    value=3,
)

# Window size control
window_size = st.number_input("Days to show:", min_value=5, max_value=200, value=DEFAULT_WINDOW_SIZE, step=5)

# Projection controls
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

# Build candlestick chart
fig_top = go.Figure(
    data=[go.Candlestick(
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
                df["Open"], df["High"], df["Low"], df["Close"]
            )
        ],
    )]
)

# Initial ranges
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
fig_top.update_xaxes(range=[start_idx, end_idx])
window_slice = df.iloc[start_idx:end_idx + 1]
w_low, w_high = float(window_slice["Low"].min()), float(window_slice["High"].max())
span, pad = max(1e-6, w_high - w_low), 0.05 * (w_high - w_low)
fig_top.update_yaxes(range=[w_low - pad, w_high + pad])
tick_step = max(1, (end_idx - start_idx) // 10)
tick_vals = list(range(start_idx, end_idx + 1, tick_step))
tick_text = [df["Date"].iloc[i].strftime("%Y-%m-%d") for i in tick_vals]
fig_top.update_layout(xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=45))

# Cache helpers
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

# ===== Overlay projections =====
if st.session_state.get("show_proj"):
    overlay_min, overlay_max, overlay_end_idx = None, None, end_idx
    pos_prices_by_k, neg_prices_by_k = {}, {}
    last_idx = len(df) - 1

    n_recent = int(recent_bars)
    sel_dates = df["Date"].iloc[max(0, len(df) - n_recent):].dt.strftime("%Y-%m-%d").tolist()
    colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#1A73E8", "#C084FC", "#00C853"]

    dtw = 0.3
    for i, sel_date in enumerate(sel_dates):
        color = colors[i % len(colors)]
        # Locate selected date index and close (make sure we use integer position)
        try:
            sel_dt = pd.to_datetime(sel_date).normalize()
            hits = df.index[df["Date"].dt.normalize() == sel_dt].tolist()
            if not hits:
                continue
            # hits[0] is the index label; get integer position
            pattern_label = hits[0]
            pattern_pos = df.index.get_loc(pattern_label)  # integer position used for plotting
            pattern_close = float(df["Close"].iloc[pattern_pos])
        except Exception:
            continue

        # Mark the selected date on the chart (use pattern_pos)
        fig_top.add_vline(x=pattern_pos, line_width=1, line_dash="dot", line_color=color)

        patterns_data = get_patterns_data_cached(sel_date)
        if not patterns_data: continue
        matching = [(k, v) for k, v in patterns_data.items() if "_steps_" in k and v.get("occurrences_count", 0) > 0]
        if bar_count != "All":
            matching = [(k, v) for k, v in matching if str(int(k.split("_steps_")[1])) == bar_count]
        matching.sort(key=lambda kv: kv[1].get("occurrences_count", 0), reverse=True)
        matching = matching[:max_overlay]

        for key, val in matching:
            df_outcomes = compute_outcomes_cached(tuple(val.get("occurrences", [])), sel_date, DEFAULT_PROJ_DAYS)
            max_k_here = min(DEFAULT_PROJ_DAYS, len(df_outcomes))
            for k in range(1, max_k_here + 1):
                if str(k) not in df_outcomes.index: continue
                row = df_outcomes.loc[str(k)]
                dom, pos_rng, neg_rng = float(row.get("dominant_probability", 0.0)), float(row.get("positive_range", 0.0) or 0.0), float(row.get("negative_range", 0.0) or 0.0)

                if dom > 0:
                    y_lvl, proj_color = pattern_close + pos_rng, "blue"
                    pos_prices_by_k.setdefault(k, []).append(y_lvl)
                elif dom < 0:
                    y_lvl, proj_color = pattern_close + neg_rng, "red"
                    neg_prices_by_k.setdefault(k, []).append(y_lvl)
                else:
                    y_lvl, proj_color = pattern_close + (pos_rng + neg_rng) / 2.0, "grey"

                proj_idx = pattern_pos + k
                fig_top.add_trace(go.Scatter(
                    x=[proj_idx - dtw, proj_idx + dtw],
                    y=[y_lvl, y_lvl],
                    mode="lines",
                    line=dict(dash="dash", width=2, color=proj_color),
                    showlegend=False,
                    hovertemplate=f"<b>{sel_date}</b><br>P+{k}: %{y_lvl:.2f}<extra></extra>",
                ))
                overlay_min = y_lvl if overlay_min is None else min(overlay_min, y_lvl)
                overlay_max = y_lvl if overlay_max is None else max(overlay_max, y_lvl)
                overlay_end_idx = max(overlay_end_idx, proj_idx)

    # ===== Aggregated line using adaptive clustering =====
    def get_adaptive_cluster_mean(points, base_gap=0.3, max_gap=1.5):
        """
        Find the mean of the largest cluster in points with an adaptive gap.
        - base_gap: minimum allowed distance between points in a cluster
        - max_gap: maximum allowed distance between points in a cluster
        """
        if not points:
            return None
        
        points_sorted = sorted(points)
        n_points = len(points_sorted)
        
        # Dynamically adjust gap based on the spread of points
        if n_points > 1:
            spread = points_sorted[-1] - points_sorted[0]
            cluster_gap = max(base_gap, min(spread / (n_points - 1), max_gap))
        else:
            cluster_gap = base_gap
        
        clusters = []
        current_cluster = [points_sorted[0]]
        
        for p in points_sorted[1:]:
            if abs(p - current_cluster[-1]) <= cluster_gap:
                current_cluster.append(p)
            else:
                clusters.append(current_cluster)
                current_cluster = [p]
        clusters.append(current_cluster)
        
        # Find the largest cluster
        largest_cluster = max(clusters, key=lambda c: len(c), default=None)
        if largest_cluster is None:
            return None
        return mean(largest_cluster)

    # Merge all positive and negative prices for clustering
    all_prices_by_k = {}
    all_keys = sorted(set(pos_prices_by_k.keys()) | set(neg_prices_by_k.keys()))
    for k in all_keys:
        combined_points = pos_prices_by_k.get(k, []) + neg_prices_by_k.get(k, [])
        cluster_mean = get_adaptive_cluster_mean(combined_points, base_gap=0.2, max_gap=1.5)  # tweak thresholds here
        if cluster_mean is not None:
            all_prices_by_k[k] = cluster_mean

    # Build the aggregated line
    if all_prices_by_k:
        x_agg = [last_idx + k for k in sorted(all_prices_by_k.keys())]
        y_agg = [all_prices_by_k[k] for k in sorted(all_prices_by_k.keys())]
        fig_top.add_trace(go.Scatter(
            x=x_agg,
            y=y_agg,
            mode="lines+markers",
            name="Aggregated Cluster Line",
            line=dict(width=3, color="#00E5FF")
        ))

    # ===== Simple mean aggregated positive/negative lines =====
    if pos_prices_by_k:
        x_pos = [last_idx + k for k in sorted(pos_prices_by_k.keys())]
        y_pos = [mean(pos_prices_by_k[k]) for k in sorted(pos_prices_by_k.keys())]
        fig_top.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode="lines+markers",
            name="Mean Positive",
            line=dict(width=3, color="#00C853"),
            connectgaps=True,
        ))
    if neg_prices_by_k:
        x_neg = [last_idx + k for k in sorted(neg_prices_by_k.keys())]
        y_neg = [mean(neg_prices_by_k[k]) for k in sorted(neg_prices_by_k.keys())]
        fig_top.add_trace(go.Scatter(
            x=x_neg,
            y=y_neg,
            mode="lines+markers",
            name="Mean Negative",
            line=dict(width=3, color="#FF3D00"),
            connectgaps=True,
        ))


    if overlay_min is not None and overlay_max is not None:
        new_low, new_high = min(w_low, overlay_min), max(w_high, overlay_max)
        pad2 = 0.05 * (new_high - new_low)
        fig_top.update_yaxes(range=[new_low - pad2, new_high + pad2])
    if overlay_end_idx > end_idx:
        fig_top.update_xaxes(range=[start_idx, overlay_end_idx])

# Show chart
st.plotly_chart(fig_top, use_container_width=True, key="main_candles_chart")

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
