"""
Chart and visualization utilities
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional


def build_projection_chart(
    df_full: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    pattern_date: str,
    look_ahead: int = 30,
    real_candles: int = 30,
    highlights: dict | None = None
) -> go.Figure:
    """
    Single chart per pattern using index-based plotting to eliminate gaps:
      - Uses index for x-axis to create continuous chart without gaps
      - Maps dates back to actual trading dates for display
      - Projections anchor at the SELECTED date, not the latest bar.
      - Initial x-view focuses on last `real_candles` BEFORE the selected date + `look_ahead` projected trading days.
      - For k=1..look_ahead, draw dashed horizontal segment at:
          pattern_close + positive_max[k] if dom=+1 (blue)
          pattern_close + negative_max[k] if dom=-1 (red)
          pattern_close if dom=0 (gray)
      - Auto-scale Y to that window only.
    """
    df_full = df_full.copy()

    # normalize selected date, find its index
    pattern_date_dt = pd.to_datetime(pattern_date).normalize()
    hits = df_full.index[df_full["Date"] == pattern_date_dt].tolist()
    if not hits:
        # safety: if UI bypassed our earlier check
        raise ValueError(f"Selected date {pattern_date} not found in dataset.")
    pattern_idx = hits[0]
    pattern_close = float(df_full["Close"].iloc[pattern_idx])

    fig = go.Figure()

    # Create index-based x-axis for continuous plotting
    df_full['Index'] = df_full.index
    
    # Add all candlesticks using index for x-axis (no gaps)
    fig.add_trace(go.Candlestick(
        x=df_full["Index"],
        open=df_full["Open"],
        high=df_full["High"],
        low=df_full["Low"],
        close=df_full["Close"],
        name="SPY",
        hoverinfo="text",
        hovertext=[f"Date: {date}<br>Open: {op}<br>High: {hi}<br>Low: {lo}<br>Close: {cl}"
                   for date, op, hi, lo, cl in zip(
                       df_full["Date"].dt.strftime("%Y-%m-%d"),
                       df_full["Open"],
                       df_full["High"],
                       df_full["Low"],
                       df_full["Close"]
                   )]
    ))

    # Optional: thin marker for selected date (unobtrusive)
    fig.add_vline(
        x=pattern_idx,
        line_width=1,
        line_dash="dot",
        line_color="white"
    )

    # Build projections on actual trading dates following the selected date
    proj_days = min(look_ahead, len(outcomes_df))
    proj_levels = []
    for k in range(1, proj_days + 1):
        row = outcomes_df.loc[str(k)] if str(k) in outcomes_df.index else None
        if row is None:
            dom = 0.0
            pos_max = 0.0
            neg_max = 0.0
        else:
            dom = float(row.get("dominant_probability", 0.0))
            pos_max = row.get("positive_range", 0.0)
            neg_max = row.get("negative_range", 0.0)
            pos_max = float(pos_max) if pd.notna(pos_max) else 0.0
            neg_max = float(neg_max) if pd.notna(neg_max) else 0.0

        if dom > 0:
            y_lvl = pattern_close + pos_max
            color = "blue"
        elif dom < 0:
            y_lvl = pattern_close + neg_max
            color = "red"
        else:
            y_lvl = pattern_close
            color = "gray"

        proj_levels.append(y_lvl)

        # P+K placed on continuous index positions (no jump at end of history)
        proj_idx = pattern_idx + k

        # Draw a short horizontal dashed segment centered at proj_idx
        dtw = 0.3  # dash width in index units
        fig.add_trace(go.Scatter(
            x=[proj_idx - dtw, proj_idx + dtw],
            y=[y_lvl, y_lvl],
            mode="lines",
            line=dict(dash="dash", width=3, color=color),
            name=f"P+{k}",
            showlegend=False,
            hovertemplate=f"<b>Projection Day {k}</b><br>" +
                         f"Price Level: %{{y:.2f}}<br>" +
                         f"<extra></extra>"
        ))

    # Add highlight points at D5/D10/.../D30 if provided
    if highlights:
        for k, entry in highlights.items():
            dom = entry.get("dom", 0)
            val = float(entry.get("value", 0.0) or 0.0)
            proj_idx = pattern_idx + int(k)
            # compute y: base on pattern_close plus range effect
            if dom == 0:
                y_pt = pattern_close
                marker_color = "#FFFFFF"  # white
            elif dom > 0:
                y_pt = pattern_close + val
                marker_color = "#00E5FF"  # bright cyan
            else:
                y_pt = pattern_close + val
                marker_color = "#FF00FF"  # magenta

            fig.add_trace(go.Scatter(
                x=[proj_idx],
                y=[y_pt],
                mode="markers",
                marker=dict(size=10, color=marker_color, symbol="star", line=dict(color="#000", width=1)),
                name=f"D{k} highlight",
                showlegend=False,
                hovertemplate=f"<b>Highlight D{k}</b><br>Dom: {dom}<br>Value: %{{y:.2f}}<extra></extra>"
        ))

    # Initial x-window: last `real_candles` BEFORE selected date + projected trading dates after
    start_view = max(0, pattern_idx - real_candles + 1)
    # Calculate end index to include projections seamlessly
    if pattern_idx + proj_days < len(df_full):
        end_idx = pattern_idx + proj_days
    else:
        # Continue projections from the last real data point
        end_idx = len(df_full) - 1 + proj_days

    fig.update_xaxes(range=[start_view, end_idx])

    # Auto-scale Y to that window only
    window_slice = df_full.iloc[start_view:pattern_idx + 1]
    if not window_slice.empty:
        window_low = float(window_slice["Low"].min())
        window_high = float(window_slice["High"].max())
    else:
        window_low = float(df_full["Low"].iloc[pattern_idx])
        window_high = float(df_full["High"].iloc[pattern_idx])

    if proj_levels:
        window_low = min(window_low, min(proj_levels))
        window_high = max(window_high, max(proj_levels))

    span = max(1e-6, window_high - window_low)
    pad = 0.05 * span
    fig.update_yaxes(range=[window_low - pad, window_high + pad])

    # Create custom tick labels that show actual dates (no duplication beyond history)
    tick_vals = []
    tick_text = []
    step = max(1, (end_idx - start_view) // 10)
    for i in range(start_view, min(end_idx + 1, len(df_full)), step):
        tick_vals.append(i)
        tick_text.append(df_full["Date"].iloc[i].strftime("%Y-%m-%d"))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title="Projection from Selected Date (dominance-based)",
        xaxis_title="Trading Days",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=45
        )
    )

    return fig


def compute_stepwise_highlights(outcomes_df: pd.DataFrame, steps: list[int] | None = None) -> dict:
    """Compute highlight values at D5, D10, ..., D30 based on dominant_probability.

    Rules per checkpoint k:
      - if dom > 0 → pick max(positive_range[1..k])
      - if dom < 0 → pick min(negative_max[1..k]) (most negative)
      - if dom == 0 → 0.0
    Returns {k: {"dom": int, "value": float}} for k in steps intersect available days.
    """
    if steps is None:
        steps = [5, 10, 15, 20, 25, 30]

    if outcomes_df is None or outcomes_df.empty:
        return {}

    # outcomes_df index are strings of day numbers; ensure safe access
    # Build a numeric-day-indexed view
    try:
        days_numeric = [int(str(i)) for i in outcomes_df.index]
        df_num = outcomes_df.copy()
        df_num["_day"] = days_numeric
        df_num = df_num.sort_values("_day")
    except Exception:
        # fallback to original
        df_num = outcomes_df.copy()
        df_num["_day"] = range(1, len(df_num) + 1)

    highlights: dict[int, dict] = {}
    for k in steps:
        subset = df_num[df_num["_day"] <= k]
        if subset.empty:
            continue
        # dom at checkpoint k (use exact row if present, else last available before k)
        dom_row = df_num[df_num["_day"] == k]
        if dom_row.empty:
            dom_row = subset.tail(1)
        dom_val = float(dom_row.get("dominant_probability", pd.Series([0.0])).iloc[0])
        dom = 1 if dom_val > 0 else (-1 if dom_val < 0 else 0)

        value: float
        pos_max_val: float = 0.0
        neg_max_val: float = 0.0
        
        if dom > 0:
            series = pd.to_numeric(subset.get("positive_range", pd.Series([0.0])), errors="coerce").fillna(0.0)
            value = float(series.max()) if not series.empty else 0.0
        elif dom < 0:
            series = pd.to_numeric(subset.get("negative_range", pd.Series([0.0])), errors="coerce").fillna(0.0)
            # choose most negative (minimum)
            value = float(series.min()) if not series.empty else 0.0
        else:
            value = 0.0

        # Always compute pos_max and neg_max for display
        pos_series = pd.to_numeric(subset.get("positive_max", pd.Series([0.0])), errors="coerce").fillna(0.0)
        neg_series = pd.to_numeric(subset.get("negative_max", pd.Series([0.0])), errors="coerce").fillna(0.0)
        pos_max_val = float(pos_series.max()) if not pos_series.empty else 0.0
        neg_max_val = float(neg_series.min()) if not neg_series.empty else 0.0

        highlights[k] = {"dom": dom, "value": value, "pos_max": pos_max_val, "neg_max": neg_max_val}

    return highlights


def render_highlights(highlights: dict):
    """Render highlights in a row of compact metrics above the table."""
    if not highlights:
        return
    ks = sorted(highlights.keys())
    cols = st.columns(len(ks))
    for idx, k in enumerate(ks):
        entry = highlights[k]
        dom = entry.get("dom", 0)
        val = entry.get("value", 0.0)
        pos_max = entry.get("pos_max", 0.0)
        neg_max = entry.get("neg_max", 0.0)
        arrow = "↑" if dom > 0 else ("↓" if dom < 0 else "→")
        label = f"D{k}"
        value_txt = f"{arrow} {val:.2f}"
        cols[idx].metric(label, value_txt)
        
        # Display pos_max and neg_max below each metric
        with cols[idx]:
            st.caption(f"Highest Pos Max: {pos_max:.2f}")
            st.caption(f"Highest Neg Max: {neg_max:.2f}")
