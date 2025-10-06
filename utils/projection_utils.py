"""
Projection Utilities
Handles projection overlay logic and pattern analysis
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean

from .chart_builder import add_vertical_marker, add_projection_line, add_aggregate_line
from .clustering_utils import cluster_projection_points, calculate_mean_aggregates, prepare_aggregate_line_data


def get_selected_dates(df: pd.DataFrame, recent_bars: int) -> List[str]:
    """
    Get the list of selected dates for projection analysis
    
    Args:
        df: DataFrame with stock data
        recent_bars: Number of recent bars to select
        
    Returns:
        List of date strings in YYYY-MM-DD format
    """
    n_recent = int(recent_bars)
    return df["Date"].iloc[max(0, len(df) - n_recent):].dt.strftime("%Y-%m-%d").tolist()


def get_pattern_position_and_close(df: pd.DataFrame, sel_date: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Get the pattern position and close price for a selected date
    
    Args:
        df: DataFrame with stock data
        sel_date: Selected date string
        
    Returns:
        Tuple of (pattern_position, pattern_close_price) or (None, None) if not found
    """
    try:
        sel_dt = pd.to_datetime(sel_date).normalize()
        hits = df.index[df["Date"].dt.normalize() == sel_dt].tolist()
        if not hits:
            return None, None
        
        pattern_label = hits[0]
        pattern_pos = df.index.get_loc(pattern_label)
        pattern_close = float(df["Close"].iloc[pattern_pos])
        return pattern_pos, pattern_close
    except Exception:
        return None, None


def process_pattern_projections(fig, df: pd.DataFrame, sel_date: str, pattern_pos: int, pattern_close: float,
                              patterns_data: Dict, bar_count: str, max_overlay: int,
                              get_patterns_data_cached, compute_outcomes_cached,
                              DEFAULT_PROJ_DAYS: int) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], 
                                                              float, float, int]:
    """
    Process pattern projections for a selected date
    
    Args:
        fig: Plotly figure to add projection lines to
        df: DataFrame with stock data
        sel_date: Selected date string
        pattern_pos: Pattern position in the dataframe
        pattern_close: Pattern close price
        patterns_data: Patterns data dictionary
        bar_count: Bar count filter
        max_overlay: Maximum overlay patterns per date
        get_patterns_data_cached: Cached function to get patterns data
        compute_outcomes_cached: Cached function to compute outcomes
        DEFAULT_PROJ_DAYS: Default projection days
        
    Returns:
        Tuple of (pos_prices_by_k, neg_prices_by_k, overlay_min, overlay_max, overlay_end_idx)
    """
    pos_prices_by_k = {}
    neg_prices_by_k = {}
    overlay_min = None
    overlay_max = None
    overlay_end_idx = pattern_pos
    dtw = 0.3
    
    # Get matching patterns
    matching = [(k, v) for k, v in patterns_data.items() if "_steps_" in k and v.get("occurrences_count", 0) > 0]
    if bar_count != "All":
        matching = [(k, v) for k, v in matching if str(int(k.split("_steps_")[1])) == bar_count]
    matching.sort(key=lambda kv: kv[1].get("occurrences_count", 0), reverse=True)
    matching = matching[:max_overlay]
    
    # Process each matching pattern
    for key, val in matching:
        df_outcomes = compute_outcomes_cached(tuple(val.get("occurrences", [])), sel_date, DEFAULT_PROJ_DAYS)
        max_k_here = min(DEFAULT_PROJ_DAYS, len(df_outcomes))
        
        for k in range(1, max_k_here + 1):
            if str(k) not in df_outcomes.index:
                continue
                
            row = df_outcomes.loc[str(k)]
            dom = float(row.get("dominant_probability", 0.0))
            pos_rng = float(row.get("positive_range", 0.0) or 0.0)
            neg_rng = float(row.get("negative_range", 0.0) or 0.0)
            
            # Determine projection level and color
            if dom > 0:
                y_lvl = pattern_close + pos_rng
                proj_color = "blue"
                pos_prices_by_k.setdefault(k, []).append(y_lvl)
            elif dom < 0:
                y_lvl = pattern_close + neg_rng
                proj_color = "red"
                neg_prices_by_k.setdefault(k, []).append(y_lvl)
            else:
                y_lvl = pattern_close + (pos_rng + neg_rng) / 2.0
                proj_color = "grey"
            
            # Add individual projection line to chart
            proj_idx = pattern_pos + k
            add_projection_line(fig, proj_idx, y_lvl, proj_color, sel_date, k)
            
            # Update overlay bounds
            overlay_min = y_lvl if overlay_min is None else min(overlay_min, y_lvl)
            overlay_max = y_lvl if overlay_max is None else max(overlay_max, y_lvl)
            overlay_end_idx = max(overlay_end_idx, proj_idx)
    
    return pos_prices_by_k, neg_prices_by_k, overlay_min, overlay_max, overlay_end_idx


def add_projection_overlays(fig, df: pd.DataFrame, sel_dates: List[str], colors: List[str],
                          bar_count: str, max_overlay: int, get_patterns_data_cached,
                          compute_outcomes_cached, DEFAULT_PROJ_DAYS: int, selected_category: str = 'All') -> Tuple[Dict[int, List[float]], 
                                                                                                                   Dict[int, List[float]], 
                                                                                                                   float, float, int]:
    """
    Add projection overlays to the chart
    
    Args:
        fig: Plotly figure to add overlays to
        df: DataFrame with stock data
        sel_dates: List of selected dates
        colors: List of colors for different dates
        bar_count: Bar count filter
        max_overlay: Maximum overlay patterns per date
        get_patterns_data_cached: Cached function to get patterns data
        compute_outcomes_cached: Cached function to compute outcomes
        DEFAULT_PROJ_DAYS: Default projection days
        
    Returns:
        Tuple of (pos_prices_by_k, neg_prices_by_k, overlay_min, overlay_max, overlay_end_idx)
    """
    all_pos_prices = {}
    all_neg_prices = {}
    global_overlay_min = None
    global_overlay_max = None
    global_overlay_end_idx = len(df) - 1
    
    for i, sel_date in enumerate(sel_dates):
        color = colors[i % len(colors)]
        
        # Get pattern position and close
        pattern_pos, pattern_close = get_pattern_position_and_close(df, sel_date)
        if pattern_pos is None or pattern_close is None:
            continue
        
        # Add vertical marker
        add_vertical_marker(fig, pattern_pos, color)
        
        # Get patterns data
        patterns_data = get_patterns_data_cached(sel_date)
        if not patterns_data:
            continue
        
        # Filter patterns by category if not 'All'
        if selected_category != 'All':
            # Get all occurrence dates for this date
            all_occurrences = []
            for key, val in patterns_data.items():
                if val.get("occurrences_count", 0) > 0:
                    all_occurrences.extend(val.get("occurrences", []))
            
            # Group the dates and filter by category
            try:
                from .technical_utils import group_days
                groups = group_days(all_occurrences, df)
                category_dates = groups.get(selected_category, [])
                
                # Filter patterns_data to only include patterns that have occurrences in the selected category
                filtered_patterns_data = {}
                for key, val in patterns_data.items():
                    if val.get("occurrences_count", 0) > 0:
                        # Check if any of this pattern's occurrences are in the selected category
                        pattern_occurrences = val.get("occurrences", [])
                        if any(date in category_dates for date in pattern_occurrences):
                            # Filter the occurrences to only include those in the selected category
                            filtered_occurrences = [date for date in pattern_occurrences if date in category_dates]
                            filtered_patterns_data[key] = {
                                **val,
                                "occurrences": filtered_occurrences,
                                "occurrences_count": len(filtered_occurrences)
                            }
                
                patterns_data = filtered_patterns_data
            except Exception as e:
                # If grouping fails, continue with all patterns
                pass
        
        # Process projections
        pos_prices, neg_prices, overlay_min, overlay_max, overlay_end_idx = process_pattern_projections(
            fig, df, sel_date, pattern_pos, pattern_close, patterns_data, bar_count, max_overlay,
            get_patterns_data_cached, compute_outcomes_cached, DEFAULT_PROJ_DAYS
        )
        
        # Merge with global collections
        for k, prices in pos_prices.items():
            all_pos_prices.setdefault(k, []).extend(prices)
        for k, prices in neg_prices.items():
            all_neg_prices.setdefault(k, []).extend(prices)
        
        # Update global overlay bounds
        if overlay_min is not None:
            global_overlay_min = overlay_min if global_overlay_min is None else min(global_overlay_min, overlay_min)
        if overlay_max is not None:
            global_overlay_max = overlay_max if global_overlay_max is None else max(global_overlay_max, overlay_max)
        global_overlay_end_idx = max(global_overlay_end_idx, overlay_end_idx)
    
    return all_pos_prices, all_neg_prices, global_overlay_min, global_overlay_max, global_overlay_end_idx


def add_aggregate_lines(fig, pos_prices_by_k: Dict[int, List[float]], 
                       neg_prices_by_k: Dict[int, List[float]], last_idx: int) -> None:
    """
    Add aggregate lines to the chart
    
    Args:
        fig: Plotly figure to add lines to
        pos_prices_by_k: Dictionary mapping projection day to positive price points
        neg_prices_by_k: Dictionary mapping projection day to negative price points
        last_idx: Last index of the data
    """
    # Add clustered aggregate line
    clustered_prices = cluster_projection_points(pos_prices_by_k, neg_prices_by_k)
    if clustered_prices:
        x_agg, y_agg = prepare_aggregate_line_data(clustered_prices, last_idx)
        add_aggregate_line(fig, x_agg, y_agg, "Aggregated Cluster Line", "#00E5FF", 3)
    
    # Add mean positive line
    if pos_prices_by_k:
        pos_aggregates = calculate_mean_aggregates(pos_prices_by_k, {})['positive']
        x_pos, y_pos = prepare_aggregate_line_data(pos_aggregates, last_idx)
        add_aggregate_line(fig, x_pos, y_pos, "Mean Positive", "#00C853", 3, True)
    
    # Add mean negative line
    if neg_prices_by_k:
        neg_aggregates = calculate_mean_aggregates({}, neg_prices_by_k)['negative']
        x_neg, y_neg = prepare_aggregate_line_data(neg_aggregates, last_idx)
        add_aggregate_line(fig, x_neg, y_neg, "Mean Negative", "#FF3D00", 3, True)
