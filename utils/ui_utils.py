"""
UI Utilities
Handles UI components, session state management, and user interactions
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

# Import other utils at the top to avoid circular imports
from .pattern_utils import analyze_current_day_pattern_outcomes, generate_cache
from .chart_utils import build_projection_chart, compute_stepwise_highlights, render_highlights
from .technical_utils import group_days, format_group_display, get_group_summary_stats

# Conditional import for AWS utils (requires boto3)
try:
    from .aws_utils import cache_exists_in_s3, load_cache_from_s3
except ImportError:
    # AWS utils not available (boto3 not installed)
    def cache_exists_in_s3(*args, **kwargs):
        raise ImportError("AWS utils not available - boto3 not installed")
    def load_cache_from_s3(*args, **kwargs):
        raise ImportError("AWS utils not available - boto3 not installed")


def initialize_session_state() -> None:
    """Initialize session state variables"""
    if "show_proj" not in st.session_state:
        st.session_state["show_proj"] = False
    if "bar_count" not in st.session_state:
        st.session_state["bar_count"] = "All"


def create_pattern_filter() -> str:
    """
    Create pattern filter UI component
    
    Returns:
        Selected bar count filter
    """
    st.subheader("🔍 Pattern Filter")
    bar_count = st.selectbox(
        "Filter by number of bars:",
        options=["All", "2", "3", "4", "5", "6", "7"],
        index=0,
        help="Select the number of bars to filter pattern mappers.",
    )
    st.session_state["bar_count"] = bar_count
    return bar_count


def create_chart_controls() -> tuple:
    """
    Create chart control UI components
    
    Returns:
        Tuple of (max_overlay, window_size, recent_bars, show_proj_clicked, selected_category)
    """
    # Global category selection
    st.markdown("### 🎯 Category Selection")
    categories = ['All', '7D+', '7D-', 'Tech+', 'Tech-']
    selected_category = st.radio(
        "Choose category for projections and analysis:",
        options=categories,
        horizontal=True,
        key="global_category_selection"
    )
    
    # Max overlay patterns control
    max_overlay = st.slider(
        "Max overlay patterns per date",
        min_value=1,
        max_value=10,
        value=3,
    )
    
    # Window size control
    from config import DEFAULT_WINDOW_SIZE
    window_size = st.number_input("Days to show:", min_value=5, max_value=200, value=DEFAULT_WINDOW_SIZE, step=5)
    
    # Projection controls
    proj_col1, proj_col2, proj_col3 = st.columns([2, 2, 2])
    with proj_col1:
        show_proj_clicked = st.button("Show projections", type="primary")
        if show_proj_clicked:
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
    
    with proj_col3:
        show_regression_clicked = st.button("Show Regression Channels", type="secondary")
        if show_regression_clicked:
            st.session_state["show_regression"] = not st.session_state.get("show_regression", False)
    
    return max_overlay, window_size, recent_bars, show_proj_clicked, selected_category, show_regression_clicked


def create_pattern_analysis_ui(df: pd.DataFrame) -> tuple:
    """
    Create pattern analysis UI components
    
    Args:
        df: DataFrame with stock data
        
    Returns:
        Tuple of (pattern_date, bar_count, analyze_clicked)
    """
    st.header("🧩 Pattern Analysis")
    last_date = df["Date"].iloc[-1]
    pattern_date = st.date_input(
        "Select date for pattern analysis:",
        value=last_date
    ).strftime("%Y-%m-%d")
    
    bar_count = st.session_state.get("bar_count", "All")
    
    # Add analyze button
    analyze_clicked = st.button("📊 Analyze Patterns", type="primary")
    
    return pattern_date, bar_count, analyze_clicked


def display_pattern_analysis_results(patterns_data: Dict, bar_count: str, pattern_date: str,
                                   selected_category: str, df: pd.DataFrame, DEFAULT_PROJ_DAYS: int, DEFAULT_REAL_BARS: int) -> None:
    """
    Display pattern analysis results
    
    Args:
        patterns_data: Dictionary containing pattern data
        bar_count: Bar count filter
        pattern_date: Selected pattern date
        selected_category: Selected category to analyze
        df: DataFrame with stock data
        DEFAULT_PROJ_DAYS: Default projection days
        DEFAULT_REAL_BARS: Default real bars
    """
    
    # Initialize session state for pattern analysis if not exists
    analysis_key = f"pattern_analysis_{pattern_date}_{bar_count}_{selected_category}"
    if analysis_key not in st.session_state:
        st.session_state[analysis_key] = {
            'patterns_data': patterns_data,
            'df': df,
            'DEFAULT_PROJ_DAYS': DEFAULT_PROJ_DAYS,
            'DEFAULT_REAL_BARS': DEFAULT_REAL_BARS,
            'selected_category': selected_category,
            'groups_cache': {},
            'outcomes_cache': {}
        }
    
    # Use cached data to prevent reloads
    cached_data = st.session_state[analysis_key]
    patterns_data = cached_data['patterns_data']
    df = cached_data['df']
    DEFAULT_PROJ_DAYS = cached_data['DEFAULT_PROJ_DAYS']
    DEFAULT_REAL_BARS = cached_data['DEFAULT_REAL_BARS']
    selected_category = cached_data['selected_category']
    
    st.markdown(f"### 📊 Analyzing: {selected_category} Category")
    
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
            # Get the occurrence dates for this pattern
            occurrence_dates = val.get("occurrences", [])
            
            # Get groups for this specific pattern to show individual metrics
            pattern_groups = {}
            try:
                pattern_groups = group_days(occurrence_dates, df)
            except Exception as e:
                st.warning(f"Could not analyze market conditions for this pattern: {str(e)}")
            
            # Display individual pattern metrics
            st.markdown("#### 📊 Pattern Metrics")
            total_all = len(occurrence_dates)
            total_7d_plus = len(pattern_groups.get('7D+', []))
            total_7d_minus = len(pattern_groups.get('7D-', []))
            total_tech_plus = len(pattern_groups.get('Tech+', []))
            total_tech_minus = len(pattern_groups.get('Tech-', []))
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("All", total_all)
            with col2:
                st.metric("7D+", total_7d_plus)
            with col3:
                st.metric("7D-", total_7d_minus)
            with col4:
                st.metric("Tech+", total_tech_plus)
            with col5:
                st.metric("Tech-", total_tech_minus)
            
            # Determine which dates to analyze based on selected category
            if selected_category == 'All':
                dates_to_analyze = occurrence_dates
                category_name = "All Patterns"
            else:
                dates_to_analyze = pattern_groups.get(selected_category, [])
                category_name = f"{selected_category} Patterns"
            
            # Display category-specific analysis
            if dates_to_analyze:
                st.markdown(f"#### 📈 {category_name} ({len(dates_to_analyze)} occurrences)")
                
                # Check if category outcomes are cached
                outcomes_cache_key = f"outcomes_{key}_{selected_category}_{pattern_date}"
                if outcomes_cache_key in cached_data.get('outcomes_cache', {}):
                    category_outcomes = cached_data['outcomes_cache'][outcomes_cache_key]
                else:
                    # Analyze outcomes for selected category dates
                    category_outcomes = analyze_current_day_pattern_outcomes(
                        df, dates_to_analyze, pattern_date, look_ahead=DEFAULT_PROJ_DAYS
                    )
                    # Cache the outcomes
                    if 'outcomes_cache' not in cached_data:
                        cached_data['outcomes_cache'] = {}
                    cached_data['outcomes_cache'][outcomes_cache_key] = category_outcomes
                
                # Highlights row
                highlights = compute_stepwise_highlights(category_outcomes)
                render_highlights(highlights)
                
                # Display dataframe
                st.dataframe(category_outcomes.T)
                
                # Display projection chart
                fig_proj = build_projection_chart(
                    df_full=df,
                    outcomes_df=category_outcomes,
                    pattern_date=pattern_date,
                    look_ahead=DEFAULT_PROJ_DAYS,
                    real_candles=DEFAULT_REAL_BARS,
                    highlights=highlights
                )
                st.plotly_chart(fig_proj, use_container_width=True, key=f"proj_chart_{key}_{selected_category}")
                
            else:
                st.info(f"No dates found for {category_name}")
                if selected_category != 'All':
                    st.markdown("**Note:** This category has no pattern occurrences in the current dataset.")
    
    # Show filtering results
    if bar_count != "All":
        st.info(f"🔍 Showing {filtered_patterns} patterns with {bar_count} bars (out of {total_patterns} total patterns)")
    else:
        st.info(f"📊 Showing all {total_patterns} patterns")


def handle_cache_generation(pattern_date: str, df: pd.DataFrame) -> Optional[str]:
    """
    Handle cache generation for pattern analysis
    
    Args:
        pattern_date: Selected pattern date
        df: DataFrame with stock data
        
    Returns:
        Cache filename or None if generation failed
    """
    
    filename = f"{pattern_date}_patterns_cache.json"
    
    if not cache_exists_in_s3(filename):
        with st.spinner(f"Generating pattern cache for {pattern_date}..."):
            filename = generate_cache(df, current_date=pattern_date)
            if filename is None:
                st.error(f"❌ The selected date ({pattern_date}) was not a trading day.")
                return None
            else:
                st.success(f"✅ Cache generated in S3: {filename}")
    else:
        st.info(f"Using existing cache file from S3: {filename}")
    
    return filename


def show_analysis_prompt() -> None:
    """Show prompt for pattern analysis"""
    st.info("👆 Click 'Analyze Patterns' button to start pattern analysis")


def clear_analysis_cache(pattern_date: str, bar_count: str, selected_category: str) -> None:
    """
    Clear analysis cache when parameters change
    
    Args:
        pattern_date: Current pattern date
        bar_count: Current bar count filter
        selected_category: Selected category
    """
    analysis_key = f"pattern_analysis_{pattern_date}_{bar_count}_{selected_category}"
    if analysis_key in st.session_state:
        del st.session_state[analysis_key]
