"""
Chart Builder Utilities
Handles the construction of candlestick charts and basic chart configuration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Optional


def build_base_candlestick_chart(df: pd.DataFrame, window_size: int) -> Tuple[go.Figure, int, int, int]:
    """
    Build the base candlestick chart with proper configuration
    
    Args:
        df: DataFrame with stock data
        window_size: Number of days to show in the window
        
    Returns:
        Tuple of (figure, start_idx, end_idx, max_index)
    """
    # Build candlestick chart
    fig = go.Figure(
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
    
    # Calculate ranges
    max_index = len(df) - 1
    start_idx = max(0, max_index - int(window_size) + 1)
    end_idx = max_index
    
    # Configure layout
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Trading Days",
        yaxis_title="Price (USD)",
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        dragmode="pan",
        uirevision="keep",
    )
    
    # Set initial ranges
    fig.update_xaxes(range=[start_idx, end_idx])
    window_slice = df.iloc[start_idx:end_idx + 1]
    w_low, w_high = float(window_slice["Low"].min()), float(window_slice["High"].max())
    span, pad = max(1e-6, w_high - w_low), 0.05 * (w_high - w_low)
    fig.update_yaxes(range=[w_low - pad, w_high + pad])
    
    # Configure x-axis ticks
    tick_step = max(1, (end_idx - start_idx) // 10)
    tick_vals = list(range(start_idx, end_idx + 1, tick_step))
    tick_text = [df["Date"].iloc[i].strftime("%Y-%m-%d") for i in tick_vals]
    fig.update_layout(xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=45))
    
    return fig, start_idx, end_idx, max_index


def update_chart_ranges(fig: go.Figure, start_idx: int, end_idx: int, 
                       overlay_min: Optional[float] = None, overlay_max: Optional[float] = None,
                       overlay_end_idx: Optional[int] = None, df: Optional[pd.DataFrame] = None) -> None:
    """
    Update chart ranges to accommodate overlays
    
    Args:
        fig: Plotly figure to update
        start_idx: Starting index for x-axis
        end_idx: Ending index for x-axis
        overlay_min: Minimum y-value from overlays
        overlay_max: Maximum y-value from overlays
        overlay_end_idx: Maximum x-value from overlays
        df: DataFrame for calculating original ranges
    """
    if overlay_min is not None and overlay_max is not None and df is not None:
        # Update y-axis range to include overlays
        window_slice = df.iloc[start_idx:end_idx + 1]
        w_low, w_high = float(window_slice["Low"].min()), float(window_slice["High"].max())
        new_low, new_high = min(w_low, overlay_min), max(w_high, overlay_max)
        pad2 = 0.05 * (new_high - new_low)
        fig.update_yaxes(range=[new_low - pad2, new_high + pad2])
    
    if overlay_end_idx is not None and overlay_end_idx > end_idx:
        # Update x-axis range to include overlays
        fig.update_xaxes(range=[start_idx, overlay_end_idx])


def add_vertical_marker(fig: go.Figure, x_position: int, color: str, 
                       line_width: int = 1, line_dash: str = "dot") -> None:
    """
    Add a vertical marker line to the chart
    
    Args:
        fig: Plotly figure to add marker to
        x_position: X position for the marker
        color: Color of the marker line
        line_width: Width of the line
        line_dash: Dash pattern of the line
    """
    fig.add_vline(x=x_position, line_width=line_width, line_dash=line_dash, line_color=color)


def add_projection_line(fig: go.Figure, x_pos: int, y_level: float, color: str,
                       sel_date: str, projection_day: int, line_width: int = 2,
                       dash_width: float = 0.3) -> None:
    """
    Add a projection line to the chart
    
    Args:
        fig: Plotly figure to add line to
        x_pos: X position for the line
        y_level: Y level for the line
        color: Color of the line
        sel_date: Selected date for hover info
        projection_day: Projection day number
        line_width: Width of the line
        dash_width: Width of the dash
    """
    fig.add_trace(go.Scatter(
        x=[x_pos - dash_width, x_pos + dash_width],
        y=[y_level, y_level],
        mode="lines",
        line=dict(dash="dash", width=line_width, color=color),
        showlegend=False,
        hovertemplate=f"<b>{sel_date}</b><br>P+{projection_day}: %{y_level:.2f}<extra></extra>",
    ))


def add_aggregate_line(fig: go.Figure, x_values: list, y_values: list, 
                      name: str, color: str, line_width: int = 3,
                      connect_gaps: bool = True) -> None:
    """
    Add an aggregate line to the chart
    
    Args:
        fig: Plotly figure to add line to
        x_values: X coordinates for the line
        y_values: Y coordinates for the line
        name: Name for the legend
        color: Color of the line
        line_width: Width of the line
        connect_gaps: Whether to connect gaps in the line
    """
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        name=name,
        line=dict(width=line_width, color=color),
        connectgaps=connect_gaps,
    ))


def add_regression_channel(fig: go.Figure, channel_data: dict, pattern_pos: int) -> None:
    """
    Add regression channel lines to the chart with multiple bands (1σ, 2σ, 3σ, 4σ)
    
    Args:
        fig: Plotly figure to add channel to
        channel_data: Dictionary containing regression channel data
        pattern_pos: Position of the pattern date
    """
    if not channel_data:
        return
    
    # Get lookback and future data
    lookback_trend = channel_data['lookback_trend']
    future_trend = channel_data['future_trend']
    
    # Calculate x positions
    lookback_start = pattern_pos - len(lookback_trend) + 1
    lookback_x = list(range(lookback_start, pattern_pos + 1))
    future_x = list(range(pattern_pos + 1, pattern_pos + 1 + len(future_trend)))
    
    # Combine lookback and future data
    all_x = lookback_x + future_x
    all_trend = lookback_trend + future_trend
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=all_x,
        y=all_trend,
        mode="lines",
        name="Regression Trend",
        line=dict(width=2, color="#FFD700", dash="solid"),
        showlegend=True,
    ))
    
    # Define band colors and styles for different sigma levels
    band_configs = [
        {'level': 1, 'color': '#FF6B6B', 'width': 2, 'opacity': 0.3, 'name': '±1σ'},
        {'level': 2, 'color': '#4ECDC4', 'width': 1.5, 'opacity': 0.2, 'name': '±2σ'},
        {'level': 3, 'color': '#FFD93D', 'width': 1, 'opacity': 0.15, 'name': '±3σ'},
        {'level': 4, 'color': '#C084FC', 'width': 0.8, 'opacity': 0.1, 'name': '±4σ'},
    ]
    
    # Add multiple bands
    for config in band_configs:
        level = config['level']
        
        # Get band data
        lookback_upper = channel_data[f'lookback_upper_{level}']
        lookback_lower = channel_data[f'lookback_lower_{level}']
        future_upper = channel_data[f'future_upper_{level}']
        future_lower = channel_data[f'future_lower_{level}']
        
        # Combine lookback and future
        all_upper = lookback_upper + future_upper
        all_lower = lookback_lower + future_lower
        
        # Add upper band
        fig.add_trace(go.Scatter(
            x=all_x,
            y=all_upper,
            mode="lines",
            name=f"Upper {config['name']}",
            line=dict(width=config['width'], color=config['color'], dash="dash"),
            showlegend=True,
        ))
        
        # Add lower band
        fig.add_trace(go.Scatter(
            x=all_x,
            y=all_lower,
            mode="lines",
            name=f"Lower {config['name']}",
            line=dict(width=config['width'], color=config['color'], dash="dash"),
            showlegend=True,
        ))
        
        # Add channel fill (only for 1σ band to avoid overlapping)
        if level == 1:
            fig.add_trace(go.Scatter(
                x=all_x + all_x[::-1],
                y=all_upper + all_lower[::-1],
                fill='tonexty',
                fillcolor=f'rgba(255, 107, 107, {config["opacity"]})',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo="skip",
            ))
