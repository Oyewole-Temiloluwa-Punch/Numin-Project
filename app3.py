"""
SPY Pattern Analysis Dashboard - Refactored Main Streamlit Application (August Data)
"""

import streamlit as st
import pandas as pd
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
    build_base_candlestick_chart,
    update_chart_ranges,
    add_projection_overlays,
    add_aggregate_lines,
    initialize_session_state,
    create_pattern_filter,
    create_chart_controls,
    create_pattern_analysis_ui,
    display_pattern_analysis_results,
    handle_cache_generation,
    show_analysis_prompt
)
from config import DATA_DIR, CSV_FILENAME_SEPTEMBER, DEFAULT_PROJ_DAYS, DEFAULT_REAL_BARS, DEFAULT_WINDOW_SIZE

# Page configuration
st.set_page_config(page_title="SPY Pattern Mapper — Per-Pattern Projections (August Data)", layout="wide")

# Load data
@st.cache_data
def load_stock_data():
    """Load stock data with caching"""
    data_path = Path(__file__).resolve().parent / DATA_DIR / CSV_FILENAME_AUGUST
    return load_data(data_path)

# Initialize session state
initialize_session_state()

# Load data
df = load_stock_data()
df["Index"] = df.index

# Main title
st.title("📈 SPY — Pattern Analysis Dashboard")

# ==== TOP: Pure Candlestick Chart (no projections) ====
st.subheader("Candlestick Chart")

# Create UI components
bar_count = create_pattern_filter()
max_overlay, window_size, recent_bars, show_proj_clicked, selected_category, show_regression_clicked = create_chart_controls()

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

# ==== Main Chart Metrics ====
if st.session_state.get("show_proj"):
    st.subheader("📊 Projection Metrics")
    
    # Get selected dates for metrics calculation
    from utils.projection_utils import get_selected_dates
    sel_dates = get_selected_dates(df, recent_bars)
    
    # Calculate metrics for each category
    all_occurrences = []
    category_metrics = {"All": 0, "7D+": 0, "7D-": 0, "Tech+": 0, "Tech-": 0}
    
    for sel_date in sel_dates:
        patterns_data = get_patterns_data_cached(sel_date)
        if patterns_data:
            for key, val in patterns_data.items():
                if val.get("occurrences_count", 0) > 0:
                    # Apply bar count filter
                    if bar_count != "All" and "_steps_" in key:
                        steps = int(key.split("_steps_")[1])
                        if str(steps) != bar_count:
                            continue
                    
                    occurrences = val.get("occurrences", [])
                    all_occurrences.extend(occurrences)
                    category_metrics["All"] += len(occurrences)
    
    # Group occurrences by category
    if all_occurrences:
        try:
            from utils.technical_utils import group_days
            groups = group_days(all_occurrences, df)
            category_metrics["7D+"] = len(groups.get("7D+", []))
            category_metrics["7D-"] = len(groups.get("7D-", []))
            category_metrics["Tech+"] = len(groups.get("Tech+", []))
            category_metrics["Tech-"] = len(groups.get("Tech-", []))
        except Exception as e:
            st.warning(f"Could not calculate category metrics: {str(e)}")
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("All", category_metrics["All"])
    with col2:
        st.metric("7D+", category_metrics["7D+"])
    with col3:
        st.metric("7D-", category_metrics["7D-"])
    with col4:
        st.metric("Tech+", category_metrics["Tech+"])
    with col5:
        st.metric("Tech-", category_metrics["Tech-"])
    
    # Show summary
    st.caption(f"📈 Showing projections for {len(sel_dates)} recent bars with {bar_count} bar patterns")

# Build base candlestick chart
fig_top, start_idx, end_idx, max_index = build_base_candlestick_chart(df, window_size)

# ===== Overlay projections =====
if st.session_state.get("show_proj"):
    # Get selected dates and colors
    from utils.projection_utils import get_selected_dates
    sel_dates = get_selected_dates(df, recent_bars)
    colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#1A73E8", "#C084FC", "#00C853"]
    
    # Add projection overlays
    pos_prices_by_k, neg_prices_by_k, overlay_min, overlay_max, overlay_end_idx = add_projection_overlays(
        fig_top, df, sel_dates, colors, bar_count, max_overlay, 
        get_patterns_data_cached, compute_outcomes_cached, DEFAULT_PROJ_DAYS, selected_category
    )
    
    # Add aggregate lines
    add_aggregate_lines(fig_top, pos_prices_by_k, neg_prices_by_k, max_index)
    
    # Add regression channels if requested (only for the latest bar)
    if st.session_state.get("show_regression", False):
        from utils.technical_utils import calculate_regression_channel
        from utils.chart_builder import add_regression_channel
        
        # Use only the latest bar (last date in the data)
        latest_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")
        channel_data = calculate_regression_channel(df, latest_date)
        if channel_data:
            # Get pattern position
            pattern_pos = channel_data['pattern_position']
            # Add regression channel to chart
            add_regression_channel(fig_top, channel_data, pattern_pos)
    
    # Update chart ranges to accommodate overlays
    update_chart_ranges(fig_top, start_idx, end_idx, overlay_min, overlay_max, overlay_end_idx, df)

# Show chart
st.plotly_chart(fig_top, use_container_width=True, key="main_candles_chart")

# ==== Pattern Analysis ====
pattern_date, bar_count, analyze_clicked = create_pattern_analysis_ui(df)

if analyze_clicked:
    # Clear any existing analysis cache when new analysis is requested
    from utils.ui_utils import clear_analysis_cache
    clear_analysis_cache(pattern_date, bar_count, selected_category)
    
    # Handle cache generation
    filename = handle_cache_generation(pattern_date, df)
    if filename is None:
        st.stop()
    
    # Load patterns data
    patterns_data = load_cache_from_s3(filename)
    
    # Display pattern analysis results
    display_pattern_analysis_results(patterns_data, bar_count, pattern_date, selected_category, df, DEFAULT_PROJ_DAYS, DEFAULT_REAL_BARS)
else:
    show_analysis_prompt()
